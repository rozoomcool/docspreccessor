import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
import ollama


MODEL_NAME = "qwen3:8b"


# ===============================
# Session State Initialization
# ===============================

def init_session_state() -> None:
    """Initialize all required keys in session_state."""
    if "schema_fields" not in st.session_state:
        # Each field: {"name": str, "type": "string"/"number"/"boolean", "required": bool, "description": str}
        st.session_state.schema_fields: List[Dict[str, Any]] = []

    if "templates" not in st.session_state:
        # name -> schema dict
        st.session_state.templates: Dict[str, Dict[str, Any]] = {}

    if "processed_docs" not in st.session_state:
        # List of dicts with metadata, raw text, json_result, dataframe
        st.session_state.processed_docs: List[Dict[str, Any]] = []

    if "selected_doc_id" not in st.session_state:
        st.session_state.selected_doc_id: Optional[str] = None

    if "show_result_modal" not in st.session_state:
        st.session_state.show_result_modal: bool = False


# ===============================
# JSON Schema Builder Logic
# ===============================

def build_json_schema(fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate JSON schema of fixed shape from field definitions."""
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    for field in fields:
        name = field.get("name", "").strip()
        field_type = field.get("type", "string")
        is_required = bool(field.get("required", False))

        if not name:
            # Ignore empty names in final schema
            continue

        properties[name] = {"type": field_type}
        if is_required:
            required.append(name)

    schema: Dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return schema


def validate_schema_fields(fields: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Basic validation for fields before saving template."""
    names = [f.get("name", "").strip() for f in fields if f.get("name", "").strip()]
    if not names:
        return False, "Добавьте хотя бы одно поле с непустым именем."

    if len(set(names)) != len(names):
        return False, "Имена полей должны быть уникальными."

    return True, None


def render_schema_builder() -> None:
    """Render UI for JSON schema builder."""
    st.subheader("Structured Output Builder")

    st.markdown("**Базовая структура:** `array` из `object` без вложенных объектов.")

    add_col, _ = st.columns([1, 4])
    if add_col.button("➕ Добавить поле", key="add_field"):
        st.session_state.schema_fields.append(
            {
                "name": f"field_{len(st.session_state.schema_fields) + 1}",
                "type": "string",
                "required": False,
                "description": "",
            }
        )

    # Render fields
    for idx, field in enumerate(st.session_state.schema_fields):
        with st.container():
            # Backward compatibility for existing session_state without descriptions
            if "description" not in field:
                field["description"] = ""

            cols = st.columns([3, 2, 1, 1])
            field_name = cols[0].text_input(
                "Имя поля",
                value=field["name"],
                key=f"field_name_{idx}",
            )
            field_type = cols[1].selectbox(
                "Тип",
                options=["string", "number", "boolean"],
                index=["string", "number", "boolean"].index(field["type"]),
                key=f"field_type_{idx}",
            )
            is_required = cols[2].checkbox(
                "required",
                value=field["required"],
                key=f"field_required_{idx}",
            )
            remove = cols[3].button("🗑️", key=f"remove_field_{idx}")

            field_description = st.text_area(
                "Описание (подсказка для LLM, когда по названию непонятно)",
                value=field.get("description", ""),
                key=f"field_description_{idx}",
                height=68,
                placeholder="Например: «Сумма к оплате в рублях», «Дата счета в формате YYYY-MM-DD», «ИНН поставщика»",
            )

            # Update state
            field["name"] = field_name
            field["type"] = field_type
            field["required"] = is_required
            field["description"] = field_description

            if remove:
                st.session_state.schema_fields.pop(idx)
                st.experimental_rerun()

    schema = build_json_schema(st.session_state.schema_fields)

    st.markdown("**Сгенерированная JSON-схема:**")
    st.json(schema)

    st.markdown("---")
    st.subheader("Сохранить шаблон")

    template_name = st.text_input("Имя шаблона", key="template_name_input")
    if st.button("💾 Сохранить шаблон"):
        is_valid, error = validate_schema_fields(st.session_state.schema_fields)
        if not is_valid:
            st.error(error)
            return

        name = template_name.strip()
        if not name:
            st.error("Имя шаблона не может быть пустым.")
            return

        if name in st.session_state.templates:
            st.error("Шаблон с таким именем уже существует. Выберите другое имя.")
            return

        # Сохраняем и "чистую" JSON-схему, и метаданные полей (включая описания),
        # но описания НЕ попадают в schema, а идут только в prompt.
        fields_meta = [
            {
                "name": f.get("name", "").strip(),
                "type": f.get("type", "string"),
                "required": bool(f.get("required", False)),
                "description": (f.get("description") or "").strip(),
            }
            for f in st.session_state.schema_fields
            if f.get("name", "").strip()
        ]

        st.session_state.templates[name] = {
            "schema": schema,
            "fields": fields_meta,
        }
        st.success(f"Шаблон «{name}» сохранён.")


# ===============================
# Document Text Extraction
# ===============================

def extract_text_from_file(uploaded_file) -> str:
    """Detect file type by extension / MIME and extract text."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    elif filename.endswith(".csv"):
        return extract_text_from_tabular(uploaded_file, file_type="csv")
    elif filename.endswith((".xls", ".xlsx")):
        return extract_text_from_tabular(uploaded_file, file_type="excel")
    else:
        raise ValueError("Неподдерживаемый формат файла.")


def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages_text).strip()


def extract_text_from_docx(uploaded_file) -> str:
    # python-docx expects a file-like object
    doc = Document(uploaded_file)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(uploaded_file) -> str:
    content = uploaded_file.read()
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")


def extract_text_from_tabular(uploaded_file, file_type: str) -> str:
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "excel":
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Неподдерживаемый тип таблицы.")

    # Represent table as readable text
    return df.to_string(index=False)


# ===============================
# JSON Validation Against Schema
# ===============================

def validate_against_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Minimal JSON validation:
    - top-level array
    - each item is object
    - required fields present
    - no unexpected types (very basic check)
    """
    if not isinstance(data, list):
        return False, "Ожидается массив (`type: array`) на верхнем уровне."

    item_schema = schema.get("items", {})
    properties = item_schema.get("properties", {})
    required_fields = item_schema.get("required", [])

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Элемент с индексом {idx} не является объектом."

        # Required fields
        for req in required_fields:
            if req not in item or item[req] in (None, ""):
                return False, f"Элемент {idx}: отсутствует или пустое required-поле '{req}'."

        # Optional: basic type checks
        for key, value in item.items():
            if key not in properties:
                # Допускаем дополнительные поля, но можно запретить при желании
                continue

            expected_type = properties[key].get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Поле '{key}' в элементе {idx} должно быть строкой."
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return False, f"Поле '{key}' в элементе {idx} должно быть числом."
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False, f"Поле '{key}' в элементе {idx} должно быть булевым."

    return True, None


# ===============================
# LLM Interaction (Ollama) – Structured Extraction
# ===============================

def build_extraction_prompt(
    schema: Dict[str, Any],
    text: str,
    fields_meta: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build a deterministic prompt for structured extraction.
    Focus: Document → Text → JSON Array (matching schema).
    """
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    # Optional field hints: берём описания только из полей-конструктора,
    # не встраивая их в саму JSON-схему.
    hints_lines: List[str] = []
    if fields_meta:
        for f in fields_meta:
            name = (f.get("name") or "").strip()
            desc = (f.get("description") or "").strip()
            if not name or not desc:
                continue
            ftype = f.get("type", "string")
            hints_lines.append(f"- {name} ({ftype}): {desc}")

    hints_block = ""
    if hints_lines:
        hints_block = "\n\nОПИСАНИЯ ПОЛЕЙ (как понимать/что извлекать):\n" + "\n".join(hints_lines)

    prompt = f"""
Ты – система извлечения структурированных данных из документа.

ТВОЯ ЗАДАЧА:
1. Проанализировать текст документа.
2. Извлечь данные строго в формате JSON, который соответствует следующей JSON-схеме.
3. Вернуть ТОЛЬКО валидный JSON без каких-либо пояснений, комментариев или текста до/после.

JSON-схема (формат результата):

{schema_str}
{hints_block}

ТРЕБОВАНИЯ:
- Верхний уровень результата: всегда JSON-массив (`type: "array"`).
- Каждый элемент массива – объект (`type: "object"`).
- Поля должны соответствовать `properties` и `required` из схемы.
- Если данные отсутствуют, верни пустой массив: [].
- НЕ ДОБАВЛЯЙ полей, которых нет в схеме.
- НЕ ПИШИ НИКАКОГО ТЕКСТА, КРОМЕ JSON.

ТЕКСТ ДОКУМЕНТА ДЛЯ АНАЛИЗА:

{text}
""".strip()

    return prompt


def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.0) -> str:
    """
    Call Ollama locally using the python client.
    Returns raw string content from the model.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
        # format="json"  # Можно включить для более строгого JSON-вывода, если модель поддерживает
    )
    # For chat API, response structure: {"message": {"role": "...", "content": "..."}}
    return response["message"]["content"]


def extract_structured_data(
    schema: Dict[str, Any],
    text: str,
    fields_meta: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 1,
) -> Tuple[Optional[Any], str, Optional[str]]:
    """
    Call LLM to extract structured data.
    - Returns (parsed_json_or_None, raw_output, validation_error_or_None).
    - Performs up to (max_retries + 1) attempts if JSON invalid or schema validation fails.
    """
    base_prompt = build_extraction_prompt(schema, text, fields_meta=fields_meta)
    last_raw_output: str = ""
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        if attempt == 0:
            prompt = base_prompt
        else:
            # On retry, reinforce JSON-only requirement
            prompt = base_prompt + """

ВАЖНО:
Предыдущий ответ был невалидным JSON.
Сейчас верни ТОЛЬКО валидный JSON, строго соответствующий схеме.
Не добавляй никаких комментариев или текста вне JSON.
""".strip()

        raw_output = call_ollama(prompt)
        last_raw_output = raw_output

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            last_error = f"Ошибка JSONDecode: {e}"
            continue

        is_valid, error = validate_against_schema(parsed, schema)
        if is_valid:
            return parsed, raw_output, None

        last_error = error

    # If we are here, all attempts failed
    return None, last_raw_output, last_error


# ===============================
# LLM Interaction (Ollama) – Document Analysis / Summarization
# ===============================

def build_summary_prompt(
    text: str,
    focus: Optional[str] = None,
    level: str = "balanced",
) -> str:
    """
    Build prompt for non-structured document analysis / summarization.
    level: "short" | "balanced" | "detailed"
    """
    if level == "short":
        length_req = "Сделай очень краткое резюме (3–5 предложений)."
    elif level == "detailed":
        length_req = "Сделай подробный структурированный обзор с ключевыми пунктами и выводами."
    else:
        length_req = "Сделай сбалансированное резюме (5–10 предложений) с самыми важными фактами."

    focus_text = ""
    if focus and focus.strip():
        focus_text = f"\nДОПОЛНИТЕЛЬНЫЙ ФОКУС АНАААЛИЗА:\n{focus.strip()}\n"

    prompt = f"""
Ты – помощник по анализу документов.

ТВОЯ ЗАДАЧА:
1. Проанализировать текст документа.
2. Выделить ключевые мысли, факты, риски и выводы.
3. Сформировать читабельное резюме на русском языке.

ТРЕБОВАНИЯ К ФОРМАТУ:
- {length_req}
- Не пиши ничего, кроме самого резюме.
- Используй маркированные или нумерованные списки, если это помогает лучше структурировать ответ.{focus_text}
ТЕКСТ ДОКУМЕНТА:

{text}
""".strip()

    return prompt


def summarize_document(
    text: str,
    focus: Optional[str] = None,
    level: str = "balanced",
) -> str:
    """High-level helper: build summary prompt and call Ollama."""
    prompt = build_summary_prompt(text, focus=focus, level=level)
    # Для суммаризации допускаем небольшую креативность
    return call_ollama(prompt, model=MODEL_NAME, temperature=0.2)


# ===============================
# Processed Documents UI & Modal
# ===============================

def register_processed_document(
    filename: str,
    template_name: str,
    schema: Dict[str, Any],
    raw_text: str,
    json_result: Any,
) -> None:
    """Store processed document metadata in session_state."""
    doc_id = str(uuid.uuid4())
    df = pd.DataFrame(json_result) if isinstance(json_result, list) else pd.DataFrame()

    st.session_state.processed_docs.append(
        {
            "id": doc_id,
            "filename": filename,
            "template_name": template_name,
            "schema": schema,
            "raw_text": raw_text,
            "json_result": json_result,
            "dataframe": df,
        }
    )


def render_processed_documents_list() -> None:
    """Render list of processed documents with clickable items."""
    st.subheader("Обработанные документы")

    if not st.session_state.processed_docs:
        st.info("Пока нет обработанных документов.")
        return

    for doc in st.session_state.processed_docs:
        cols = st.columns([4, 3, 2])
        label = f"{doc['filename']} (шаблон: {doc['template_name']})"
        if cols[0].button(label, key=f"doc_button_{doc['id']}"):
            st.session_state.selected_doc_id = doc["id"]
            st.session_state.show_result_modal = True

        cols[1].markdown(f"**Строк:** {len(doc['dataframe'])}")
        cols[2].markdown(f"**ID:** `{doc['id'][:8]}...`")


@st.dialog("Результат извлечения")
def show_result_dialog() -> None:
    """Modal dialog with table and download buttons for selected document."""
    doc_id = st.session_state.get("selected_doc_id")
    if not doc_id:
        st.write("Документ не выбран.")
        if st.button("Закрыть"):
            st.session_state.show_result_modal = False
            st.experimental_rerun()
        return

    doc = next((d for d in st.session_state.processed_docs if d["id"] == doc_id), None)
    if not doc:
        st.write("Документ не найден.")
        if st.button("Закрыть"):
            st.session_state.show_result_modal = False
            st.experimental_rerun()
        return

    st.markdown(f"**Файл:** {doc['filename']}")
    st.markdown(f"**Шаблон:** {doc['template_name']}")

    st.markdown("**Табличный вид результатов:**")
    st.dataframe(doc["dataframe"], use_container_width=True)

    json_str = json.dumps(doc["json_result"], ensure_ascii=False, indent=2)
    csv_str = doc["dataframe"].to_csv(index=False)

    st.download_button(
        "⬇️ Скачать JSON",
        data=json_str.encode("utf-8"),
        file_name=f"{doc['filename']}.json",
        mime="application/json",
        key=f"download_json_{doc_id}",
    )

    st.download_button(
        "⬇️ Скачать CSV",
        data=csv_str.encode("utf-8"),
        file_name=f"{doc['filename']}.csv",
        mime="text/csv",
        key=f"download_csv_{doc_id}",
    )

    st.markdown("---")
    if st.button("Закрыть"):
        st.session_state.show_result_modal = False
        st.experimental_rerun()


# ===============================
# Document Upload & Processing UI
# ===============================

def render_document_processing() -> None:
    st.subheader("Document Upload & Processing")

    uploaded_file = st.file_uploader(
        "Загрузите документ",
        type=["txt", "pdf", "docx", "csv", "xlsx"],
    )

    if not st.session_state.templates:
        st.warning("Сначала создайте и сохраните хотя бы один шаблон в секции Structured Output Builder.")
        selected_template_name = None
    else:
        selected_template_name = st.selectbox(
            "Выберите шаблон для извлечения",
            options=list(st.session_state.templates.keys()),
        )

    if st.button("🚀 Обработать документ"):
        if not uploaded_file:
            st.error("Сначала загрузите файл.")
            return
        if not selected_template_name:
            st.error("Выберите шаблон.")
            return

        with st.spinner("Извлечение текста из документа..."):
            try:
                text = extract_text_from_file(uploaded_file)
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
                return

        template_obj = st.session_state.templates[selected_template_name]

        # Backward compatibility: старые шаблоны могли быть "чистой" схемой без метаданных полей.
        if isinstance(template_obj, dict) and "schema" in template_obj:
            schema = template_obj["schema"]
            fields_meta = template_obj.get("fields") or []
        else:
            schema = template_obj
            fields_meta = []

        with st.spinner("Вызов модели Ollama и извлечение структуры..."):
            json_result, raw_output, validation_error = extract_structured_data(
                schema,
                text,
                fields_meta=fields_meta,
                max_retries=1,
            )

        if json_result is None:
            st.error("Не удалось получить валидный JSON от модели.")
            if validation_error:
                st.warning(f"Ошибка валидации: {validation_error}")
            with st.expander("Показать сырой ответ модели"):
                st.code(raw_output)
            return

        st.success("Документ успешно обработан и данные извлечены.")
        register_processed_document(
            filename=uploaded_file.name,
            template_name=selected_template_name,
            schema=schema,
            raw_text=text,
            json_result=json_result,
        )


# ===============================
# Document Analysis & Summarization UI
# ===============================

def render_document_analysis() -> None:
    st.subheader("Document Analysis & Summarization")

    uploaded_file = st.file_uploader(
        "Загрузите документ для анализа",
        type=["txt", "pdf", "docx", "csv", "xlsx"],
        key="analysis_file_uploader",
    )

    level = st.selectbox(
        "Детализация резюме",
        options=[
            ("short", "Кратко (3–5 предложений)"),
            ("balanced", "Сбалансировано (5–10 предложений)"),
            ("detailed", "Подробно"),
        ],
        format_func=lambda x: x[1],
    )[0]

    focus = st.text_area(
        "Дополнительный фокус анализа (опционально)",
        help="Например: «Сфокусируйся на юридических рисках», «Сделай акцент на цифрах и метриках», "
             "«Выдели обязательства и сроки»",
    )

    if st.button("🔎 Проанализировать документ"):
        if not uploaded_file:
            st.error("Сначала загрузите файл.")
            return

        with st.spinner("Извлечение текста из документа..."):
            try:
                text = extract_text_from_file(uploaded_file)
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
                return

        if not text.strip():
            st.warning("В документе не удалось найти текст для анализа.")
            return

        with st.spinner("Выполняется анализ документа с помощью LLM..."):
            summary = summarize_document(text, focus=focus, level=level)

        st.success("Анализ завершён.")
        st.markdown("**Резюме документа:**")
        st.markdown(summary)

        with st.expander("Показать исходный текст документа"):
            st.text(text)


# ===============================
# Main App
# ===============================

def main() -> None:
    st.set_page_config(
        page_title="AI Structured Extraction Platform",
        layout="wide",
    )

    init_session_state()

    st.title("AI Structured Extraction Platform")
    st.caption("Document → Text → LLM → JSON Array → Table → Modal")

    builder_tab, processing_tab, analysis_tab = st.tabs(
        [
            "1. Structured Output Builder",
            "2. Document Upload & Processing",
            "3. Document Analysis & Summarization",
        ]
    )

    with builder_tab:
        render_schema_builder()

    with processing_tab:
        col_left, col_right = st.columns([2, 3])
        with col_left:
            render_document_processing()
        with col_right:
            render_processed_documents_list()

    with analysis_tab:
        render_document_analysis()

    # Show result modal if needed
    if st.session_state.get("show_result_modal", False):
        show_result_dialog()


if __name__ == "__main__":
    main()