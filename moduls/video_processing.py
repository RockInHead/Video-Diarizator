from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from moduls import helpers_diaraize as helpers
import logging, os, re, subprocess
import faster_whisper
import torch
import torchaudio


def start_extract_audio(video: str, audio_dir_path: str) -> str:
    """
    Извлекает аудио из видеофайла с использованием локального ffmpeg.exe.

    :param video_path: Путь к видеофайлу.
    :return: Путь к извлеченному аудиофайлу или None при ошибке.
    """
    # Имя файла без расширения
    video_file_name = os.path.splitext(os.path.basename(video))[0]

    # Получаем системный диск (обычно C:\)
    os.makedirs(audio_dir_path, exist_ok=True)

    # Путь до выходного аудиофайла
    audio_path = os.path.join(audio_dir_path, f"{video_file_name}.mp3")

    # Путь к локальному ffmpeg.exe
    ffmpeg_path = os.path.abspath(os.path.join("tools", "ffmpeg.exe"))

    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg.exe не найден по пути: {ffmpeg_path}")

    try:
        subprocess.run([
            ffmpeg_path,
            "-i", video,
            "-vn",  # без видео
            "-acodec", "libmp3lame",
            "-y",  # overwrite
            audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError:
        print("❌ Ошибка при извлечении аудио с помощью ffmpeg")
        return None

    return audio_path


mtypes = {"cpu": "int8", "cuda": "float16"}


def start_diarize(audio, no_stem=True, suppress_numerals=False, model_name="medium.en", 
                  batch_size=8, language=None, device=None):
    
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    language = helpers.process_language_arg(language, model_name)

    if no_stem:
        # Isolate vocals from the rest of the audio
        return_code = subprocess.run([
            "demucs",
            "-n", "htdemucs",
            "--two-stems=vocals",
            audio,
            "-o", helpers.GLOBAL_AUDIO_DIR,
            "--device", device
        ])

        if return_code != 0:
            logging.warning("Source splitting failed, using original audio file. "
                            "Use --no-stem argument to disable it."
                            )
            vocal_target = audio
        else:
            vocal_target = os.path.join(
                                        helpers.GLOBAL_AUDIO_DIR,
                                        "htdemucs",
                                        os.path.splitext(os.path.basename(audio))[0],
                                        "vocals.wav"
                                        )
    else: vocal_target = audio

    print(1)
    # Transcribe the audio file
    whisper_model = faster_whisper.WhisperModel(model_name, device=device, compute_type=mtypes[device])
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    suppress_tokens = (helpers.find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
                       if suppress_numerals
                       else [-1])

    if batch_size > 0: transcript_segments, info = whisper_pipeline.transcribe(
                                                                               audio_waveform,
                                                                               language,
                                                                               suppress_tokens=suppress_tokens,
                                                                               batch_size = batch_size,
                                                                              )
    
    else: transcript_segments, info = whisper_model.transcribe(
                                                               audio_waveform,
                                                               language,
                                                               suppress_tokens=suppress_tokens,
                                                               vad_filter=True,
                                                              )
    # так можно получить текст на этом этапе
    # full_transcript = "".join(segment.text for segment in transcript_segments)
    print(2)
    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()
    print(3)
    # Заменяем CTC forced alignment на простой mapping из faster-whisper сегментов
    word_timestamps = []
    for segment in transcript_segments:
        word_timestamps.append({
            "text": segment.text.strip(),
            "start": int(segment.start * 1000),
            "end": int(segment.end * 1000),
            "speaker": "unknown"
        })

    # convert audio to mono for NeMo combatibility
    temp_path = helpers.GLOBAL_YAML_DIR
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        torch.from_numpy(audio_waveform).unsqueeze(0).float(),
        16000,
        channels_first=True,
    )
    print(4)
    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=helpers.create_config(temp_path)).to(device)
    msdd_model.diarize()
    del msdd_model
    torch.cuda.empty_cache()
    print(5)
    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
    wsm = helpers.get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    print(6)
    if info.language in helpers.punct_model_langs:
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

    else: logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )
    print(7)
    # wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = helpers.get_sentences_speaker_mapping(wsm, speaker_ts)

    # Записываем транскрипцию в файл
    diarized_text = helpers.get_speaker_aware_transcript(ssm)
    helpers.cleanup(temp_path)
    print(8)
    return diarized_text


def extraction_text(
    video: str,
    model_name: str = "large",
    language = "ru",
) -> str:
    """
    Точка входа для пайплайна извлечения текста из видео.
    Извлекает аудио, выполняет диаризацию, восстанавливает текст с таймингами и спикерами.

    :param video: Путь к видеофайлу.
    :param no_stem: Флаг, указывающий, нужно ли извлекать вокал.
    :param suppress_numerals: Удалять ли цифры из текста.
    :param model_name: Название модели whisper.
    :param batch_size: Размер батча.
    :param language: Язык аудио.
    :param device: Устройство для запуска моделей.
    :return: Полный текст с таймингами и указанием спикеров.
    """
    # 1. Извлечение аудио из видео
    audio_path = start_extract_audio(video, helpers.GLOBAL_AUDIO_DIR)

    # Выполняем диаризацию. Сохраняем путь до файл с диаризацией диалога.
    diarized_text = start_diarize(audio = audio_path, model_name=model_name, language=language)

    print("COOL!!!", diarized_text)
    return diarized_text
