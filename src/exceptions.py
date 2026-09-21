"""利用者向けメッセージと開発者向け詳細を分離した例外。"""

from __future__ import annotations


class RAGApplicationError(Exception):
    """画面に安全に表示できるメッセージを持つ基底例外。"""

    def __init__(self, user_message: str, technical_detail: str = "") -> None:
        super().__init__(technical_detail or user_message)
        self.user_message = user_message
        self.technical_detail = technical_detail or user_message


class ConfigurationError(RAGApplicationError):
    """設定ファイルが不正な場合。"""


class UnsupportedDocumentError(RAGApplicationError):
    """未対応形式の資料が指定された場合。"""


class DocumentReadError(RAGApplicationError):
    """資料を読み取れない場合。"""


class EmptyDocumentError(RAGApplicationError):
    """検索に利用できる本文がない場合。"""


class OllamaConnectionError(RAGApplicationError):
    """ローカルOllamaへ接続できない場合。"""


class OllamaModelError(RAGApplicationError):
    """必要なOllamaモデルが利用できない場合。"""


class EmbeddingError(RAGApplicationError):
    """埋め込み生成に失敗した場合。"""


class VectorStoreError(RAGApplicationError):
    """検索データベースの操作に失敗した場合。"""


class DocumentRegistrationError(RAGApplicationError):
    """資料登録全体に失敗した場合。"""


class RetrievalError(RAGApplicationError):
    """関連資料の検索に失敗した場合。"""


class GenerationError(RAGApplicationError):
    """ローカルLLMの回答生成に失敗した場合。"""
