import argparse
import logging
import os
import re
import faster_whisper
import torch
import torchaudio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from diarizator.helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mtypes = {"cpu": "int8", "cuda": "float16"}

def write_transcript_to_file(ssm, text_file_name, text_dir_path):
    """
    Записывает транскрипцию в файл.

    :param ssm: Список предложений с информацией о спикерах.
    :param output_path: Путь до файла, куда будет записан текст.
    """
    output_path = os.path.join(text_dir_path, f"{text_file_name}.txt")
    
    with open(output_path, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)  # Запись транскрипции в файл
    return output_path
        

def start_diarize(audio, no_stem=True, suppress_numerals=False, model_name="medium.en", batch_size=8, language=None, device=None, text_dir_path = os.path.dirname(os.path.abspath(__file__))):
    
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    language = process_language_arg(language, model_name)

    if no_stem:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o temp_outputs --device "{device}"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = audio
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(audio))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio


    # Transcribe the audio file
    whisper_model = faster_whisper.WhisperModel(
       model_name, device=device, compute_type=mtypes[device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size = batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # convert audio to mono for NeMo combatibility
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        torch.from_numpy(audio_waveform).unsqueeze(0).float(),
        16000,
        channels_first=True,
    )


    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )

    # wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    #     write_srt(ssm, srt)
        
    # Записываем транскрипцию в файл
    text_file_name = os.path.splitext(os.path.basename(audio))[0]
    diarized_text_path = write_transcript_to_file(ssm, text_file_name, text_dir_path)

    cleanup(temp_path)
    return diarized_text_path
