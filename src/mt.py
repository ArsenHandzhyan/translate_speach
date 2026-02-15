"""Machine Translation engines with IT glossary support."""
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


# IT/Dev glossary: terms that should NOT be translated literally.
# Format: {lowercase_term: {lang_code: correct_translation}}
# These are applied as post-processing after MT.
IT_GLOSSARY_EN_RU = {
    # Frameworks & Libraries
    "spring": "Spring",
    "spring boot": "Spring Boot",
    "spring framework": "Spring Framework",
    "spring mvc": "Spring MVC",
    "spring security": "Spring Security",
    "spring data": "Spring Data",
    "swing": "Swing",
    "javafx": "JavaFX",
    "hibernate": "Hibernate",
    "react": "React",
    "angular": "Angular",
    "vue": "Vue",
    "node.js": "Node.js",
    "express": "Express",
    "django": "Django",
    "flask": "Flask",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "jenkins": "Jenkins",
    "gradle": "Gradle",
    "maven": "Maven",
    "kafka": "Kafka",
    "redis": "Redis",
    "nginx": "Nginx",
    "terraform": "Terraform",
    "ansible": "Ansible",

    # Java concepts
    "method overloading": "перегрузка методов",
    "method overriding": "переопределение методов",
    "overloading": "перегрузка",
    "overriding": "переопределение",
    "polymorphism": "полиморфизм",
    "encapsulation": "инкапсуляция",
    "inheritance": "наследование",
    "abstraction": "абстракция",
    "interface": "интерфейс",
    "abstract class": "абстрактный класс",
    "singleton": "синглтон",
    "dependency injection": "внедрение зависимостей",
    "garbage collection": "сборка мусора",
    "garbage collector": "сборщик мусора",
    "multithreading": "многопоточность",
    "thread pool": "пул потоков",
    "deadlock": "дедлок",
    "race condition": "состояние гонки",
    "lambda expression": "лямбда-выражение",
    "stream api": "Stream API",
    "generics": "дженерики",
    "annotation": "аннотация",
    "reflection": "рефлексия",
    "serialization": "сериализация",
    "deserialization": "десериализация",
    "hashmap": "HashMap",
    "arraylist": "ArrayList",
    "linked list": "LinkedList",
    "binary tree": "бинарное дерево",
    "hash table": "хеш-таблица",
    "stack": "стек",
    "queue": "очередь",
    "heap": "куча",

    # General IT
    "backend": "бэкенд",
    "frontend": "фронтенд",
    "full stack": "фулстек",
    "microservices": "микросервисы",
    "api": "API",
    "rest api": "REST API",
    "graphql": "GraphQL",
    "ci/cd": "CI/CD",
    "devops": "DevOps",
    "agile": "Agile",
    "scrum": "Scrum",
    "sprint": "спринт",
    "pull request": "пулл-реквест",
    "merge request": "мердж-реквест",
    "code review": "код-ревью",
    "deployment": "деплой",
    "repository": "репозиторий",
    "commit": "коммит",
    "branch": "ветка",
    "pipeline": "пайплайн",
    "load balancer": "балансировщик нагрузки",
    "caching": "кэширование",
    "database": "база данных",
    "sql": "SQL",
    "nosql": "NoSQL",
    "orm": "ORM",
    "crud": "CRUD",
    "solid": "SOLID",
    "dry": "DRY",
    "design pattern": "паттерн проектирования",
    "design patterns": "паттерны проектирования",
    "refactoring": "рефакторинг",
    "unit test": "юнит-тест",
    "unit testing": "юнит-тестирование",
    "integration test": "интеграционный тест",
    "mocking": "мокирование",
    "logging": "логирование",
    "debugging": "дебаггинг",
    "profiling": "профилирование",
    "runtime": "рантайм",
    "compile time": "время компиляции",
    "jvm": "JVM",
    "jdk": "JDK",
    "sdk": "SDK",
    "ide": "IDE",
    "git": "Git",
    "github": "GitHub",
    "bitbucket": "Bitbucket",
    "jira": "Jira",
    "confluence": "Confluence",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "cloud": "облако",
}

# Reverse glossary for RU→EN: common Russian IT transliterations
IT_GLOSSARY_RU_EN = {
    "спринг": "Spring",
    "спринг бут": "Spring Boot",
    "свинг": "Swing",
    "хибернейт": "Hibernate",
    "кубернетес": "Kubernetes",
    "кубернетис": "Kubernetes",
    "докер": "Docker",
    "кафка": "Kafka",
    "редис": "Redis",
    "ноде": "Node.js",
    "реакт": "React",
    "ангулар": "Angular",
    "джанго": "Django",
    "фласк": "Flask",
    "терраформ": "Terraform",
    "дженкинс": "Jenkins",
    "грэдл": "Gradle",
    "мавен": "Maven",
    "гитхаб": "GitHub",
    "битбакет": "Bitbucket",
    "джира": "Jira",
    "перегрузка методов": "method overloading",
    "переопределение методов": "method overriding",
    "полиморфизм": "polymorphism",
    "инкапсуляция": "encapsulation",
    "наследование": "inheritance",
    "абстракция": "abstraction",
    "синглтон": "singleton",
    "дедлок": "deadlock",
    "бэкенд": "backend",
    "фронтенд": "frontend",
    "микросервисы": "microservices",
    "деплой": "deployment",
    "рефакторинг": "refactoring",
    "пайплайн": "pipeline",
}

# Wrong translations to fix (post-processing corrections)
# Google Translate often makes these mistakes with IT terms
CORRECTIONS_EN_RU = {
    "весной": "Spring",
    "весна": "Spring",
    "весне": "Spring",
    "весну": "Spring",
    "качелями": "Swing",
    "качели": "Swing",
    "качелях": "Swing",
    "пружина": "Spring",
    "пружины": "Spring",
    "пружину": "Spring",
}

CORRECTIONS_RU_EN = {
    # Common mistranslations from Russian speech
}


def apply_glossary(text: str, src_lang: str, tgt_lang: str) -> str:
    """Apply IT glossary corrections to translated text."""
    if not text:
        return text

    result = text

    if tgt_lang == "ru":
        # Fix known bad translations EN→RU
        # First, fix specific wrong translations in context
        # "nothing related to spring" → should keep Spring, not "весна"
        for wrong, correct in CORRECTIONS_EN_RU.items():
            # Case-insensitive replacement, but only for standalone words
            pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
            # Only replace if it looks like IT context (heuristic)
            if pattern.search(result):
                result = pattern.sub(correct, result)

    elif tgt_lang == "en":
        # Fix Russian transliterations → proper English IT terms
        for ru_term, en_term in IT_GLOSSARY_RU_EN.items():
            pattern = re.compile(re.escape(ru_term), re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub(en_term, result)

    return result


class NLLBTranslator:
    """Local MT using NLLB-200 via CTranslate2."""

    LANG_MAP = {
        "ru": "rus_Cyrl",
        "en": "eng_Latn",
    }

    def __init__(self, model_path: str):
        import ctranslate2
        import sentencepiece as spm

        self.model_path = Path(model_path)
        log.info(f"Loading NLLB model from {self.model_path}")

        self.translator = ctranslate2.Translator(
            str(self.model_path),
            device="cpu",
            compute_type="int8",
        )

        sp_model = self.model_path / "sentencepiece.bpe.model"
        if not sp_model.exists():
            sp_model = self.model_path / "source.spm"
        self.tokenizer = spm.SentencePieceProcessor(str(sp_model))
        log.info("NLLB model loaded.")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text.strip():
            return ""

        src_code = self.LANG_MAP.get(src_lang, src_lang)
        tgt_code = self.LANG_MAP.get(tgt_lang, tgt_lang)

        tokens = self.tokenizer.encode(text, out_type=str)
        tokens = [src_code] + tokens

        results = self.translator.translate_batch(
            [tokens],
            target_prefix=[[tgt_code]],
            beam_size=4,
            max_decoding_length=256,
        )

        output_tokens = results[0].hypotheses[0]
        if output_tokens and output_tokens[0] == tgt_code:
            output_tokens = output_tokens[1:]

        translated = self.tokenizer.decode(output_tokens)
        # Apply glossary corrections
        translated = apply_glossary(translated, src_lang, tgt_lang)
        return translated


class SimpleCloudTranslator:
    """Cloud translation using Google Translate with IT glossary post-processing."""

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import requests

        if not text.strip():
            return ""

        try:
            resp = requests.post(
                "https://translate.googleapis.com/translate_a/single",
                params={
                    "client": "gtx",
                    "sl": src_lang,
                    "tl": tgt_lang,
                    "dt": "t",
                    "q": text,
                },
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            translated = "".join(part[0] for part in data[0] if part[0])

            # Apply IT glossary corrections
            translated = apply_glossary(translated, src_lang, tgt_lang)

            return translated
        except Exception as e:
            log.warning(f"Cloud MT failed: {e}")
            return text
