# -*- coding: utf-8 -*-
"""
CrewAI com Ollama local
Script 100% offline usando Llama3.2 para Buscador, Redator e Editor
"""
import os
from crewai import LLM, Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

os.environ["SERPER_API_KEY"] = "SUA_CHAVE_API"

# -------------------------------
# Configuração do LLM Ollama local
# -------------------------------
llm_ollama = LLM(
    model="ollama/llama3.2:latest", 
    base_url="http://localhost:11434" 
)

# -------------------------------
# Inicializa ferramentas
# -------------------------------
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# -------------------------------
# Criação dos agentes
# -------------------------------

# 1 - Buscador de Conteúdo
buscador = Agent(
    role="Buscador de Conteúdo",
    goal="Buscar conteúdo online sobre o tema {tema}",
    backstory=(
        "Você está criando artigos para o LinkedIn sobre {tema}. "
        "Você fará buscas na internet, coletará informações relevantes e organizará. "
        "Seu trabalho servirá de base para o Redator de Conteúdo."
    ),
    tools=[search_tool, scrape_tool],
    llm=llm_ollama,
    verbose=True
)

# 2 - Redator de Conteúdo
redator = Agent(
    role="Redator de Conteúdo",
    goal="Escrever um texto divertido e factualmente correto para o LinkedIn sobre {tema}",
    backstory=(
        "Você está redigindo artigos para o LinkedIn sobre {tema}. "
        "Você utilizará os dados coletados pelo Buscador de Conteúdo para criar um texto interessante, divertido e correto. "
        "Deixe claro quando forem opiniões pessoais."
    ),
    tools=[search_tool, scrape_tool],
    llm=llm_ollama,
    verbose=True
)

# 3 - Editor de Conteúdo
editor = Agent(
    role="Editor de Conteúdo",
    goal="Editar um texto de LinkedIn para que tenha um tom mais informal",
    backstory=(
        "Você vai receber um texto do Redator de Conteúdo e editá-lo para o tom de voz "
        "mais informal e divertido."
    ),
    tools=[search_tool, scrape_tool],
    llm=llm_ollama,
    verbose=True
)

# -------------------------------
# Criação das tarefas
# -------------------------------

# Tarefa 1: Buscar conteúdo
buscar = Task(
    description=(
        "1. Priorize as últimas tendências, os principais atores e notícias relevantes sobre {tema}.\n"
        "2. Identifique o público alvo, considerando seus interesses e pontos de dor.\n"
        "3. Inclua palavras-chave de SEO e dados ou fontes relevantes."
    ),
    agent=buscador,
    expected_output="Um plano de tendências sobre {tema}, com palavras-chave de SEO e últimas notícias."
)

# Tarefa 2: Redigir conteúdo
redigir = Task(
    description=(
        "1. Use os dados coletados para criar um post atraente sobre {tema}.\n"
        "2. Incorpore palavras-chave de SEO de forma natural.\n"
        "3. Estruture o post de forma cativante, com uma conclusão que faça o leitor refletir."
    ),
    agent=redator,
    expected_output="Um texto de LinkedIn sobre {tema}."
)

# Tarefa 3: Editar conteúdo
editar = Task(
    description=(
        "Revisar a postagem do LinkedIn quanto a erros gramaticais."
    ),
    agent=editor,
    expected_output="Um texto de LinkedIn pronto para publicação. O texto deve estar pronto para um post, com bastante informações relevantes."
    
)

# -------------------------------
# Criação da equipe (Crew)
# -------------------------------
equipe = Crew(
    agents=[buscador, redator, editor],
    tasks=[buscar, redigir, editar],
    llm=llm_ollama,
    verbose=True
)

# -------------------------------
# Rodando o Crew
# -------------------------------
if __name__ == "__main__":
    tema_do_artigo = "Inteligência artificial vai substituir o trabalho humano?"

    try:
        resultado = equipe.kickoff(inputs={"tema": tema_do_artigo})
        print("\nResposta final:\n")
        print(resultado)
    except Exception as e:
        print("Erro ao gerar resposta:", e)
