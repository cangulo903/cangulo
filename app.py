import time
import requests
# RAG Imports
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import openai
import os
from dotenv import load_dotenv
load_dotenv()  # Carrega as variáveis do .env
openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
PORT = os.getenv('PORT')

# Função para criar um arquivo JSON com Título, ID e URL de download de todos os vídeos de Treinamento
# def store_video_info():
#-------------A FAZER-------------------------------------------------


# Função para baixar videos. Argumento: dicionário de IDs e Títulos
from requests_oauthlib import OAuth1
import requests
from moviepy.editor import * #pip install moviepy
def download_videos(dict_videos_id_title):
    # OAuth1.0 config
    # consumer_key = 'shop-api'
    # consumer_secret = 'CWMJcyp1EQxjlNGXHZBd2vaxFPdScKGi'
    # access_token = '5d52bf03-5407-41f9-923e-225237a7499a'
    # access_token_secret = '6c28efcd-8517-474d-b268-c8a8297bcf3a01a62345-1f72-4c25-900b-c0b138bd3710'
    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)

    for id, title in dict_videos_id_title.items():
        url = f'https://fluig.cixbrasil.com/content-management/api/v2/documents/{id}/stream'
        # Nome do arquivo
        print(f"Downloading file: {id} - {title}.mp4")
        video_path = f'videos\{id}.mp4'
 
        # Coletar vídeo
        response = requests.get(url, auth=auth, stream=True)
 
        # Salvar o vídeo
        if response.status_code == 200:
            if not os.path.exists(video_path):
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size = 1024*1024):
                        if chunk:
                            f.write(chunk)
                print(f"{id} - {title}.mp4 downloaded!\n")
                # Guardar áudio e deletar vídeo
                video = VideoFileClip(video_path)
                audio = video.audio
                audio_path = f"audios\{id}.mp3"
                audio.write_audiofile(audio_path)
                print(f"Áudio extraído e salvo em {audio_path}")
                time.sleep(3)
        else:
            print(f"Error: {id} - {title}.mp4 failed!\n")
        #os.remove(video_path)
        #print(f'{id} - Vídeo deletado, áudio guardado')
    return print("All audios saved!")


# Função para criar e armazenar as transcrições JÁ CARREGADAS NO VECTORSTORE
import openai
import io
def store_transcriptions(dict_videos_id_title):
    for id, title in dict_videos_id_title.items():
        # Caso o vectorstore não exista
        vectorstore_path = fr"vectorstores\vectorstore_{id}"
        print(vectorstore_path)
        if not os.path.exists(vectorstore_path):
            # Dividir o áudio em pedaços menores que o tamanho máximo da OpenAI Whisper
            chunk_size = 24 * 1024 * 1024  # 24MB
            audio_text = ""
            file_path = f"audios\{id}.mp3"

            # Obter o tamanho do arquivo
            file_size = os.path.getsize(file_path)

            with open(file_path, 'rb') as audio_file:
                # Se o arquivo for menor que 24MB, transcrever diretamente
                if file_size <= chunk_size:
                    print(f"Arquivo {file_path} com {file_size} bytes, transcrevendo diretamente.")
                    transcription = openai.Audio.transcribe(
                        model="whisper-1", 
                        file=audio_file,
                        api_key= OPENAI_API_KEY
                    )
                    audio_text += transcription['text']
                # Se o arquivo for maior, separar em chunks, pois a OpenAI só aceita até 25MB
                print(f"Arquivo {file_path} com {file_size} bytes, transcrevendo em chunks.")
                while True:
                    chunk = audio_file.read(chunk_size)
                    if not chunk:
                        break
                    # Use io.BytesIO to create an in-memory file-like object from the chunk
                    with io.BytesIO(chunk) as chunk_file:
                        chunk_file.name = os.path.basename(file_path)  # Simula um nome de arquivo
                        transcription = openai.Audio.transcribe(
                            model="whisper-1", 
                            file=chunk_file,
                            api_key= OPENAI_API_KEY
                        )
                        audio_text += transcription['text'] + ' '
            audio_text.strip()

            # Gerar o vectorstore
            openai.api_key
            # Dividir e criar os embeddings
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.create_documents([audio_text])
            # Salvar os vectorstores
            vectorstore_name = f"vectorstore_{id}"
            persist_directory = os.path.join("vectorstores", vectorstore_name)

            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
            vectorstore.persist()
            print(f"Vectorstore {id} stored!\n")

            # Deletando o áudio
            os.remove(file_path)
        else:
            print(f"vectorstore_{id} already exists")
    return print("All vectorstores stored!")


# Gerador de respostas, argumentos: questão do usuário, chave da api, vectorstore
def response_generator(user_input, id):
    # Carregar o vectorstore escolhido
    # Definir o diretório do vectorstore específico
    vectorstore_name = f'vectorstore_{id}'
    persist_directory = os.path.join("vectorstores", vectorstore_name)
    # Carregar o vectorstore salvo
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(),)
    print(f"Vectorstore '{vectorstore_name}' loaded from '{persist_directory}'!\n")

    # Buscar uma resposta relevante no conteúdo do vídeo
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # Prompt e modelo escolhido
    prompt = hub.pull("babsky/rag-prompt-pt-br")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    # Formatação
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Pega a resposta da API
    return rag_chain.invoke(user_input)
    # response = rag_chain.invoke(user_input)
    # for word in response.split():
    #     yield word + " "
    #     time.sleep(0.08)

# Dicionário IDs e Títulos dos vídeos
dict_videos_id_title = {
    # Página Fluig
    '4521': 'Como abrir uma solicitação',
    '4524': 'Como cancelar uma solicitação',
    '4529': 'Como avaliar uma solicitação',
    '4522': 'Como acompanhar uma solicitação',
    '4531': 'Como atender uma solicitação',
    '4530': 'O usuário pode devolver a solicitação ao responsável, veja o que fazer',
    # Página SAFE-DOC
    '20888': 'Como subir arquivos no SAFE-DOC',
    '20887': 'Como fazer buscas avançadas no SAFE-DOC',
    # Página SGA
    '80720': 'Atendendo uma senha',
    '80718': 'Ativação de uma senha agendada',
    '80719': 'Emissão de senha por agendamento',
    '80721': 'Emissão de senhas do dia',
    # Página Performance e Metas
    '129566': 'Performance e Metas - Feedback',
    # Página Pedido de Compras
    '457681': '1 Treinamento - Medição e Lançamento de Pré-Nota',
    '457695': '2 Treinamento - Medição e Lançamento de Pré-Nota'
}

# Função para encontrar o ID correspondente ao título input do usuario
def find_id_by_title(dict_id_title, user_option):
    for id, title in dict_id_title.items():
        if user_option == title:
            return id
    return None  # Retorna None se o título não for encontrado

# download_videos(dict_videos_id_title)
# store_transcriptions(dict_videos_id_title)

# OPERAÇÃO
current_id = None
from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins
@app.route('/process_vectorstore', methods=['POST'])
def process_vectorstore():
    global current_id
    data = request.json
    print(data)
    title = data.get('title')
    print(f'Título: {title}')
    id = find_id_by_title(dict_videos_id_title, title)
    print(f'ID: {id}')
    current_id = id
    return jsonify({'message': f'Você enviou o título {title} de ID {id}'})

@app.route('/process_questions', methods=['POST'])
def process_questions():
    global current_id
    data = request.json
    print(data)
    question = data.get('question')
    print(f'Question: {question}')
    chat_reply = response_generator(question, current_id)
    print(f'ChatReply: {chat_reply}')
    return jsonify({'response': chat_reply})

@app.route('/', methods=['GET'])
def home():
    return "OK", 200

if __name__ == '__main__':
    app.run(port=PORT)
