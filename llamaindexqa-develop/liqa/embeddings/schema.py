from enum import Enum

class EmbeddingModel(str, Enum):
    BGE_LARGE_ZH = "BAAI/bge-large-zh"
    BGE_LARGE_ZH_V1_5 = "BAAI/bge-large-zh-v1.5"
    TAO = "amu/tao"
    TAO_8K = "amu/tao-8k"
    STELLA_BASE_ZH = "infgrad/stella-base-zh"
    STELLA_BASE_ZH_V2 = "infgrad/stella-base-zh-v2"
    STELLA_LARGE_ZH = "infgrad/stella-large-zh"
    STELLA_LARGE_ZH_V2 = "infgrad/stella-large-zh-v2"
    PICCOLO_LARGE_ZH = "sensenova/piccolo-large-zh"
    GTE_BASE_ZH = "thenlper/gte-base-zh"
    GTE_LARGE_ZH = "thenlper/gte-large-zh"
    M3E='moka-ai/m3e-large'

    def __str__(self):
        return self.value
