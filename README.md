# Video-Diarizator

## Описание

**Video-Diarizator** — это инструмент для диаризации видео, который принимает на вход видеофайл и генерирует текстовый файл с расшифровкой, отражающей речь спикеров.
Получившийся файл отправляется по API в ChatGPT, который делает краткую выжимку собеседования, выделяя плюсы, минусы и область работы кандидата.

## Установка
Необходим Python >= 3.10. К сожалению версия 3.13 пока что не поддерживается.

`FFMPEG` и `Cython` необходимы в качестве предварительных условий для установки. 
Как правильно скачать `FFMPEG`, можно посмотреть [тут](https://www.youtube.com/watch?v=9_ldCQUgU7Q).

```
pip install ffmpeg-python
```
```
pip install cython
```

Также требуется предустановить Perl. У меня предустановлен Strawberry Perl, который можно установить [отсюда](https://strawberryperl.com/).

Для ChatGPT необходим API ключ, который нужно сохранить в переменную окружения.

## Установка зависимостей
```
pip install -c constraints.txt -r requirements.txt
```

## Навигация по файлам
- `main`: Главный запускаемый файл, в котором можно выбрать необходимое видео и папку сохранения для выходных данных.
- `assistant`: Файл, где происходит создание ассистента ChatGPT и отсылается запрос для анализа диаризированного файла. На выходе получаем файл .txt c краткой выжимкой плюсов, минусов и области работы кандидата.
- `diarize`: Файл, где происходит транскрибация и диаризация.
- `helpers`: Вспомогательный файл, для `diarize`.
  
