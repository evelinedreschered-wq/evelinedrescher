
from fastapi import FastAPI

# CONFIGURAÇÃO DA APLICAÇÃO


app = FastAPI(
    title="Assistente de IA para Portfólio Profissional",
    description=(
        "Assistente virtual para responder perguntas sobre "
        "minha trajetória profissional, competências e projetos."
    ),
    version="1.0.0"
)

# ROTA PRINCIPAL

@app.get("/")
def inicio():
    return {
        "mensagem": "Assistente de IA para Portfólio Profissional",
        "status": "Aplicação em funcionamento"
    }

# ROTA DE SAÚDE DA APLICAÇÃO ######

@app.get("/health")
def verificar_saude():
    return {
        "status": "online",
        "mensagem": "Backend funcionando corretamente."
    }
