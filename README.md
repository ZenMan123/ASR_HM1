# Отчет

## Ссылки на раны

- [Краткий отчет в Comet ML](https://www.comet.com/artemsafin67/pytorch-template-asr-example/reports/JjLdu8DpZZWf8ZXXjvVknvflr)
- [Бейзлайн](https://www.comet.com/artemsafin67/pytorch-template-asr-example/me9ph890x936gkyhjxyh9ohzgo8a461q?&prevPath=%2Fartemsafin67%2Fpytorch-template-asr-example%2Fview%2Fnew%2Fpanels)
- [Бим серч без LM](https://www.comet.com/artemsafin67/pytorch-template-asr-example/me9ph890x936gkyhjxyh9ohzgo8a461q?&prevPath=%2Fartemsafin67%2Fpytorch-template-asr-example%2Fview%2Fnew%2Fpanels)
- [Бим серч с LM + sentencepiece токенизатор](https://www.comet.com/artemsafin67/pytorch-template-asr-example/4mhtv4lx3v5b550xjykb0z4638uhs8zg?compareXAxis=step&experiment-tab=panels&prevPath=%2Fartemsafin67%2Fpytorch-template-asr-example%2Fview%2Fnew%2Fpanels&showOutliers=true&smoothing=0&xAxis=step)
- [Аудио аугментации](https://www.comet.com/artemsafin67/pytorch-template-asr-example/mvqvhvrd1uflt8pkpmyc2vmftc162xzp?compareXAxis=step&experiment-tab=panels&prevPath=%2Fartemsafin67%2Fpytorch-template-asr-example%2Fview%2Fnew%2Fpanels&showOutliers=true&smoothing=0&xAxis=step)

## Итоговый скор

Мой лучший чекпоинт выбивает на тестовой выборке WER 0.2974. Так же я руками реализовал beam-search, сделал прогоны beam-search + LM и заменил пословный токенизатор на SentencePiece.

## Обучение модели

В качестве модели был выбран Conformer примерно на 130M параметров. В целях упрощения реализации вместо относительного позиционного кодирования были использованы классические синусно-косинусные позиционные эмбеддинги.

Использовался конфиг `baseline.yaml` с `text_encoder=sp_ctc_text_encoder` (BPE вместо посимвольной токенизации) и `metrics=beam_lm` (beam_search + LM во время инференса). В качестве трейн датасета использовался `train-clean-100`

## Попытки выбить качество лучше

Не сказал бы, что было очень много экспериментов. Скорее у меня были итеративные изменения, каждое из которых давало прирост качества.

1. Увеличил размер модели с 30M до 130M
1. Заменил argmax декодирование на самописный beam-search
1. Самописный beam-search оказался жутко неоптимальным, переписал на библиотечное решение
1. Добавил в библиотечный beam-search вероятности из LM
1. Заменил посимвольную токенизацию на SentencePiece токенизатор

К сожалению, из-за нехватки времени не удалось провести много других интересных экспериментов. Как минимум, обучался я все время не на полном трейн сплите и совсем не пытался перебирать гиперы. Так же уверен, что можно было сделать модельку сильно меньше, при этом не теряя так много в качестве (архитектурные экспы). А еще у меня не было ни одного рана с использованием аугментаций (несмотря на это они реализованы и протестированы)

## Сложности

Основные сложности были связаны с осознанием того, как работает библиотека. После этого я просто заполнял шаблонные части, тестировал их на onebatchtest. Конечно, была пара багов, но они довольно легко искались и исправлялись после того как я изучил код библиотеки. А на нормальные эксперименты у меня уже не хватило времени :(

## Демо

Ноутбук, в котором описывается как можно замерить модельку на произвольном датасете лежит [здесь](demo.ipynb). Установка довольно тривиальная, необходимо поставить на чистое окружение все библиотеки, которые указаны в [requirements.txt](requirements.txt) и скачать чекпоинты с HuggingFace. Все эти шаги описаны в демо

## Проверка эффективности кастомного beam-search + буст от добавления LM в декодер

[Ноутбук с проверкой](beam_search_proof.ipynb)