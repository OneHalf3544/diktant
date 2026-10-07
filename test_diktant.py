"""Тесты чистой логики diktant.py: нормализация текста, разбивка на
предложения и отрезки, планирование пауз, формулировка заголовка.
Синтез, сеть, ffmpeg и файловый вывод не затрагиваются.

Запуск: python -m unittest
"""

import unittest

import diktant

MAX_WORDS = 5
MIN_WORDS = 2


class TestNormalizeDashes(unittest.TestCase):
    def test_дефис_в_пробелах_становится_тире(self):
        cases = [
            ("самый верный способ - это закрыть двери",
             "самый верный способ — это закрыть двери"),
            ("Влага, накопившаяся в земле, - надежный фундамент",
             "Влага, накопившаяся в земле, — надежный фундамент"),
            ("Диван -- тоже вещь", "Диван — тоже вещь"),
            ("- Привет, - сказал он.", "— Привет, — сказал он."),
        ]
        for src, want in cases:
            with self.subTest(src=src):
                self.assertEqual(diktant.normalize_dashes(src), want)

    def test_внутрисловный_дефис_не_затрагивается(self):
        for src in ("научно-технического", "чьи-либо", "пр-т Мира",
                    "16-20 градусов", "из 3-4 ярких цветов"):
            with self.subTest(src=src):
                self.assertEqual(diktant.normalize_dashes(src), src)


class TestSplitSentences(unittest.TestCase):
    def test_точка_делит_предложения(self):
        self.assertEqual(
            diktant.split_sentences("Он был широк в плечах. В садах цветет акация."),
            ["Он был широк в плечах.", "В садах цветет акация."],
        )

    def test_сокращение_не_конец_предложения(self):
        self.assertEqual(
            diktant.split_sentences("В 1812 г. случилась война. Об этом помнят."),
            ["В 1812 г. случилась война.", "Об этом помнят."],
        )

    def test_инициалы_не_рвут_предложение(self):
        cases = [
            ("В 1958 году Б. Л. Пастернак был удостоен премии. Об этом писали.",
             ["В 1958 году Б. Л. Пастернак был удостоен премии.", "Об этом писали."]),
            ("Начало связано с именем Ю. А. Гагарина. Он был первым.",
             ["Начало связано с именем Ю. А. Гагарина.", "Он был первым."]),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(diktant.split_sentences(text), want)

    def test_номер_пункта_остается_при_своем_тексте(self):
        self.assertEqual(
            diktant.split_sentences("1. Делу время, а потехе час. 2. Всему свое время."),
            ["1. Делу время, а потехе час.", "2. Всему свое время."],
        )

    def test_продолжение_со_строчной_буквы_приклеивается(self):
        self.assertEqual(
            diktant.split_sentences("Он сказал: «Да». и добавил тихо."),
            ["Он сказал: «Да». и добавил тихо."],
        )


class TestSplitLongSentence(unittest.TestCase):
    def split(self, sentence):
        return diktant.split_long_sentence(sentence, MAX_WORDS, MIN_WORDS)

    def test_режет_по_знакам_препинания(self):
        self.assertEqual(
            self.split("Влага, накопившаяся в земле, — надежный фундамент будущего урожая."),
            ["Влага,", "накопившаяся в земле, —", "надежный фундамент будущего урожая."],
        )

    def test_знак_остается_в_конце_отрезка(self):
        self.assertEqual(
            self.split("Хлеб да вода — богатырская еда."),
            ["Хлеб да вода —", "богатырская еда."],
        )

    def test_кусок_без_пунктуации_дробится_по_max_words(self):
        parts = self.split("Он закрыл лицо руками и долго молчал у самого окна в тишине.")
        self.assertEqual(parts, ["Он закрыл лицо руками и",
                                 "долго молчал у самого окна",
                                 "в тишине."])
        self.assertTrue(all(diktant.count_words(p) <= MAX_WORDS for p in parts))

    def test_осколок_короче_min_words_приклеивается(self):
        self.assertEqual(
            self.split("Он читает роман о мужестве, да."),
            ["Он читает роман о мужестве, да."],
        )

    def test_инициалы_не_отрываются_от_фамилии(self):
        self.assertEqual(
            self.split("В 1958 году Б. Л. Пастернак был удостоен Нобелевской премии."),
            ["В 1958 году Б. Л. Пастернак", "был удостоен Нобелевской премии."],
        )


class TestBuildSegments(unittest.TestCase):
    def test_номер_пункта_не_считается_словом(self):
        segments = diktant.build_segments("1. Делу время, а потехе час.",
                                          MAX_WORDS, MIN_WORDS)
        self.assertEqual([s.text for s in segments],
                         ["1. Делу время,", "а потехе час."])
        self.assertEqual([s.words for s in segments], [2, 3])

    def test_конец_предложения_отмечен_на_последнем_отрезке(self):
        segments = diktant.build_segments(
            "Он закрыл лицо руками. Мы взглянули друг на друга, и нас поразило подозрение.",
            MAX_WORDS, MIN_WORDS)
        self.assertEqual([s.is_sentence_end for s in segments],
                         [True, False, True])
        self.assertEqual([s.sentence_index for s in segments], [0, 1, 1])

    def test_отрезки_хранят_предложение_целиком(self):
        sentence = "Мы взглянули друг на друга, и нас поразило подозрение."
        for seg in diktant.build_segments(sentence, MAX_WORDS, MIN_WORDS):
            with self.subTest(text=seg.text):
                self.assertEqual(seg.sentence_text, sentence)


class TestPlanPauses(unittest.TestCase):
    """Темп (слова / общая длительность) — главный инвариант планировщика."""

    TEXT = ("Радость и мир в сердце каждого, кто строит, созидает. "
            "Степные дороги называют хлебными трассами. "
            "Долг каждого — помнить, что он в ответе за всё, что делается рядом.")

    SPEECH_PER_WORD = 0.4

    def plan(self, wpm, *, recap=True, repeat=1, min_pause=1.0):
        """Планирует паузы на искусственной, но детерминированной речи:
        длительность отрезка пропорциональна числу слов в нём."""
        segments = diktant.build_segments(self.TEXT, MAX_WORDS, MIN_WORDS)
        for seg in segments:
            seg.speech_sec = seg.words * self.SPEECH_PER_WORD
        groups = ([g for g in diktant.sentence_groups(segments) if len(g) >= 2]
                  if recap else [])
        for group in groups:
            group[0].recap = True
            group[0].recap_pre_pause = 1.5
            group[0].recap_speech_sec = (
                diktant.count_words(group[0].sentence_text) * self.SPEECH_PER_WORD)
        return diktant.plan_pauses(
            segments, wpm, repeat=repeat, repeat_gap=0.8, min_pause=min_pause,
            sentence_extra=1.0, clause_extra=0.4, lead_in=1.0, tail=3.0,
        )

    def test_целевой_темп_выдерживается(self):
        for wpm in (5, 10, 20):
            for recap in (True, False):
                for repeat in (1, 2):
                    with self.subTest(wpm=wpm, recap=recap, repeat=repeat):
                        plan, warning = self.plan(wpm, recap=recap, repeat=repeat)
                        self.assertIsNone(warning)
                        self.assertAlmostEqual(plan.total, plan.words / wpm * 60, places=6)
                        self.assertAlmostEqual(plan.actual_wpm, wpm, places=6)

    def test_рекап_берется_из_бюджета_а_не_сверх_него(self):
        with_recap, _ = self.plan(20, recap=True)
        without_recap, _ = self.plan(20, recap=False)
        self.assertAlmostEqual(with_recap.total, without_recap.total, places=6)
        self.assertLess(with_recap.pause_total, without_recap.pause_total)

    def test_пауза_пропорциональна_числу_слов(self):
        plan, _ = self.plan(10)
        inner = [s for s in plan.segments if not s.is_sentence_end]
        short = min(inner, key=lambda s: s.words)
        long = max(inner, key=lambda s: s.words)
        self.assertLess(short.pause_sec, long.pause_sec)

    def test_недостижимый_темп_дает_предупреждение_и_минимальные_паузы(self):
        plan, warning = self.plan(300, min_pause=1.0)
        self.assertIsNotNone(warning)
        self.assertIn("недостижим", warning)
        for seg in plan.segments:
            with self.subTest(text=seg.text):
                extra = 1.0 if seg.is_sentence_end else 0.4
                self.assertAlmostEqual(seg.pause_sec, 1.0 + extra, places=6)
        self.assertLess(plan.actual_wpm, 300)


class TestParseDuration(unittest.TestCase):
    def test_форматы(self):
        cases = [
            ("4", 240.0), ("4m", 240.0), ("4min", 240.0), (" 4 мин ", 240.0),
            ("4.5", 270.0), ("270s", 270.0), ("270 секунд", 270.0),
            ("4:30", 270), ("1:02:30", 3750),
        ]
        for spec, want in cases:
            with self.subTest(spec=spec):
                self.assertEqual(diktant.parse_duration(spec), want)

    def test_мусор_отвергается(self):
        for spec in ("", "abc", "4:ab", "1:2:3:4", "-5", "4:30:"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    diktant.parse_duration(spec)


class TestHeader(unittest.TestCase):
    def test_склонение_по_числу(self):
        cases = [(1, "слова"), (21, "слова"), (2, "слов"), (5, "слов"),
                 (11, "слов"), (111, "слов"), (175, "слов")]
        for n, want in cases:
            with self.subTest(n=n):
                self.assertEqual(diktant.plural_ru(n, "слова", "слов", "слов"), want)

    def test_дробное_число_в_родительном(self):
        self.assertEqual(diktant.plural_ru(7.5, "слово", "слова", "слов"), "слова")

    def test_длительность_словами(self):
        cases = [(0, "0 секунд"), (59, "59 секунд"), (60, "1 минута"),
                 (90, "1 минута 30 секунд"), (120, "2 минуты"),
                 (270, "4 минуты 30 секунд"), (3600, "1 час"),
                 (3661, "1 час 1 минута 1 секунда")]
        for seconds, want in cases:
            with self.subTest(seconds=seconds):
                self.assertEqual(diktant.spoken_duration(seconds), want)

    def test_заголовок_с_названием_и_автором(self):
        self.assertEqual(
            diktant.header_text("Пушкин — великий художник слова",
                                "по В. Г. Белинскому", 178, 20, None),
            "Пушкин — великий художник слова. по В. Г. Белинскому. "
            "Диктант из 178 слов. Темп 20 слов в минуту.",
        )

    def test_заголовок_с_длительностью_вместо_темпа(self):
        self.assertEqual(
            diktant.header_text("Упражнение 107", None, 154, 20, 270),
            "Упражнение 107. Диктант из 154 слов. Длительность 4 минуты 30 секунд.",
        )

    def test_дробный_темп_читается_с_запятой(self):
        self.assertEqual(diktant.header_text(None, None, 1, 7.5, None),
                         "Диктант из 1 слова. Темп 7,5 слова в минуту.")


if __name__ == "__main__":
    unittest.main()
