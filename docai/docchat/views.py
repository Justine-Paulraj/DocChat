from django.conf import settings
from django.shortcuts import render
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from .forms import DocumentUploadForm, QuestionForm
from .models import Document
import os
import requests
import tempfile

VECTORSTORE_DIR = os.path.join(settings.BASE_DIR, "vectorstores")


def load_pdf_from_cloudinary(file_url):
    """Download PDF from a remote URL (Cloudinary) and return a PyPDFLoader."""
    response = requests.get(file_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
    return PyPDFLoader(tmp_file_path)


def get_loader(doc):
    """Return a PyPDFLoader for local or Cloudinary file."""
    if doc.file.url.startswith("http"):
        # Remote file (Cloudinary)
        return load_pdf_from_cloudinary(doc.file.url)
    else:
        # Local file
        return PyPDFLoader(doc.file.path)


def get_vectorstore(doc):
    """Load or create a vectorstore for the document."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    store_path = os.path.join(VECTORSTORE_DIR, str(doc.id))

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    if os.path.exists(store_path):
        # Load existing vectorstore
        return Chroma(persist_directory=store_path, embedding_function=embeddings)
    else:
        # Create new vectorstore
        loader = get_loader(doc)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=store_path)
        vectorstore.persist()
        return vectorstore


def home(request):
    upload_form = DocumentUploadForm()
    question_form = QuestionForm()
    answer = None
    filename = request.session.get("uploaded_filename")

    # --- Upload file ---
    if request.method == "POST" and "upload_file" in request.POST:
        upload_form = DocumentUploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            uploaded_file = upload_form.cleaned_data["file"]

            doc = Document(file=uploaded_file)
            doc.save()

            filename = uploaded_file.name
            request.session["uploaded_filename"] = filename
            request.session["document_id"] = doc.id
            request.session["conversation"] = []
            request.session.modified = True

            # Create vectorstore once
            get_vectorstore(doc)

            return render(request, "docchat/upload_success.html", {
                "filename": filename,
                "question_form": question_form,
                "conversation": [],
                "answer": None
            })

    # --- Ask question ---
    elif request.method == "POST" and "ask_question" in request.POST:
        question_form = QuestionForm(request.POST)
        if not filename:
            return render(request, "docchat/upload.html", {
                "form": upload_form,
                "error": "Please upload a document first."
            })

        doc_id = request.session.get("document_id")
        doc = Document.objects.get(id=doc_id) if doc_id else None
        if not doc:
            return render(request, "docchat/upload.html", {"form": upload_form, "error": "File missing."})

        vectorstore = get_vectorstore(doc)

        if question_form.is_valid():
            question = question_form.cleaned_data["question"]
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            answer = qa.run(question)

            conversation = request.session.get("conversation", [])
            conversation.append({"question": question, "answer": answer})
            request.session["conversation"] = conversation
            request.session.modified = True

        return render(request, "docchat/upload_success.html", {
            "filename": filename,
            "question_form": QuestionForm(),
            "conversation": request.session.get("conversation", []),
        })

    # --- Reset ---
    elif request.method == "POST" and "reset" in request.POST:
        for key in ["conversation", "uploaded_filename", "document_id"]:
            request.session.pop(key, None)
        request.session.modified = True
        return render(request, "docchat/upload.html", {"form": upload_form})

    # --- Clear chat ---
    elif request.method == "POST" and "clear_chat" in request.POST:
        request.session["conversation"] = []
        request.session.modified = True
        return render(request, "docchat/upload_success.html", {
            "filename": request.session.get("uploaded_filename"),
            "question_form": QuestionForm(),
            "conversation": [],
            "answer": None
        })

    return render(request, "docchat/upload.html", {"form": upload_form})
