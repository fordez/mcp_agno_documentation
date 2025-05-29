# Documentación MCP y Agno: Creación de Agentes con Protocolo de Contexto de Modelo

## Índice

1. [Introducción al Protocolo de Contexto de Modelo (MCP)](#1-introducción-al-protocolo-de-contexto-de-modelo-mcp)
2. [Agno: Framework para Agentes de IA](#2-agno-framework-para-agentes-de-ia)
3. [Creación de Clientes MCP con Agno](#3-creación-de-clientes-mcp-con-agno)
4. [Implementación de Servidores MCP](#4-implementación-de-servidores-mcp)
5. [Descubrimiento y Registro entre Clientes y Servidores](#5-descubrimiento-y-registro-entre-clientes-y-servidores)
6. [Ejecución de Herramientas por LLMs](#6-ejecución-de-herramientas-por-llms)
7. [Guía para Construir tu Propio Agente](#7-guía-para-construir-tu-propio-agente)
8. [Ejemplos Avanzados y Casos de Uso](#8-ejemplos-avanzados-y-casos-de-uso)
9. [Solución de Problemas Comunes](#9-solución-de-problemas-comunes)
10. [Recursos Adicionales](#10-recursos-adicionales)

## 1. Introducción al Protocolo de Contexto de Modelo (MCP)

El Protocolo de Contexto de Modelo (MCP, por sus siglas en inglés) es un estándar abierto que define cómo los modelos de lenguaje (LLMs) pueden interactuar con herramientas y recursos externos. MCP permite a los LLMs descubrir, invocar y utilizar herramientas proporcionadas por servidores externos, ampliando significativamente sus capacidades.

### 1.1 Conceptos Clave de MCP

- **Cliente MCP**: Aplicación que utiliza un LLM y necesita acceder a herramientas externas.
- **Servidor MCP**: Proporciona herramientas y recursos que pueden ser utilizados por los LLMs.
- **Herramientas (Tools)**: Funciones o capacidades expuestas por los servidores MCP.
- **Transporte**: Mecanismo de comunicación entre clientes y servidores (STDIO, SSE, Streamable HTTP).

### 1.2 Arquitectura Básica de MCP

```
+----------------+        +----------------+        +----------------+
|                |        |                |        |                |
|  Cliente MCP   | <----> |      LLM       | <----> |  Servidor MCP  |
|                |        |                |        |                |
+----------------+        +----------------+        +----------------+
                                  ^
                                  |
                                  v
                          +----------------+
                          |                |
                          |  Servidor MCP  |
                          |                |
                          +----------------+
```

### 1.3 Tipos de Transporte en MCP

MCP soporta tres tipos principales de transporte:

1. **STDIO**: Comunicación a través de entrada/salida estándar, ideal para desarrollo local.
2. **SSE (Server-Sent Events)**: Comunicación basada en eventos HTTP, con conexiones persistentes.
3. **Streamable HTTP**: Transporte más moderno que permite comunicación bidireccional a través de un único endpoint HTTP.

## 2. Agno: Framework para Agentes de IA

Agno es una biblioteca ligera y de alto rendimiento para construir agentes de IA. Proporciona una interfaz unificada para trabajar con diferentes LLMs y facilita la integración con el protocolo MCP.

### 2.1 Instalación de Agno

```bash
pip install agno
```

### 2.2 Características Principales de Agno

- Soporte para múltiples proveedores de LLM (OpenAI, Anthropic, etc.)
- Integración nativa con MCP
- Capacidades de razonamiento y memoria
- Arquitectura multi-agente
- Soporte para herramientas personalizadas

### 2.3 Ejemplo Básico de Agno

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Crear un agente simple
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="Agente de asistencia general",
    markdown=True
)

# Obtener respuesta del agente
response = agent.get_response("¿Cuál es la capital de Francia?")
print(response)
```

## 3. Creación de Clientes MCP con Agno

### 3.1 Cliente MCP Básico

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

# Configurar cliente MCP
mcp_client = MCPTools(
    transport="streamable-http",  # Usar streamable HTTP como transporte
    url="http://localhost:8000/mcp",  # URL del servidor MCP
    auth_headers={"Authorization": "Bearer your-token"}  # Opcional
)

# Crear agente con herramientas MCP
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[mcp_client],
    description="Agente con acceso a herramientas MCP",
    markdown=True
)

# Usar el agente con herramientas MCP
response = agent.get_response("Analiza los datos de ventas del último trimestre")
print(response)
```

### 3.2 Cliente MCP con Múltiples Servidores

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

# Configurar múltiples clientes MCP
data_analysis_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8001/mcp",
    name="data_analysis"  # Nombre para identificar este servidor
)

text_processing_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8002/mcp",
    name="text_processing"
)

image_generation_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8003/mcp",
    name="image_generation"
)

# Crear agente con múltiples servidores MCP
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[data_analysis_mcp, text_processing_mcp, image_generation_mcp],
    description="Agente multimodal con múltiples servidores MCP",
    markdown=True
)

# El agente puede usar herramientas de cualquiera de los servidores
response = agent.get_response(
    "Analiza los datos de ventas, resume los hallazgos y genera una imagen de un gráfico de barras"
)
print(response)
```

### 3.3 Cliente MCP con Cambio Dinámico de LLM

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools

# Configurar cliente MCP
mcp_tools = MCPTools(
    transport="streamable-http",
    url="http://localhost:8000/mcp"
)

# Crear diferentes instancias de LLM
gpt4_model = OpenAIChat(id="gpt-4o")
claude_model = Claude(id="claude-3-opus-20240229")

# Función para cambiar de LLM dinámicamente
def switch_llm(agent, model_name):
    if model_name == "gpt4":
        agent.model = gpt4_model
    elif model_name == "claude":
        agent.model = claude_model
    return f"Cambiado a modelo: {model_name}"

# Crear agente inicial con GPT-4
agent = Agent(
    model=gpt4_model,
    tools=[mcp_tools],
    description="Agente con capacidad de cambio de LLM",
    markdown=True
)

# Usar el agente con GPT-4
response_gpt4 = agent.get_response("Explica la teoría de la relatividad")
print(f"Respuesta de GPT-4: {response_gpt4}")

# Cambiar a Claude
switch_llm(agent, "claude")

# Usar el agente con Claude
response_claude = agent.get_response("Explica la teoría de la relatividad")
print(f"Respuesta de Claude: {response_claude}")
```

## 4. Implementación de Servidores MCP

### 4.1 Servidor MCP Básico con FastMCP

```python
from fastapi import FastAPI
from fastmcp import MCPServer, Tool, Resource

app = FastAPI()
mcp_server = MCPServer()

# Definir una herramienta simple
@mcp_server.tool
def suma(a: int, b: int) -> int:
    """Suma dos números enteros.
    
    Args:
        a: Primer número
        b: Segundo número
        
    Returns:
        La suma de a y b
    """
    return a + b

# Definir un recurso (información estática)
@mcp_server.resource
def información_empresa() -> dict:
    """Proporciona información sobre la empresa."""
    return {
        "nombre": "Ejemplo Corp",
        "fundación": 2020,
        "empleados": 150,
        "ubicación": "Madrid, España"
    }

# Montar el servidor MCP en FastAPI
app.mount("/mcp", mcp_server.app)

# Ejecutar con: uvicorn nombre_archivo:app --host 0.0.0.0 --port 8000
```

### 4.2 Servidor MCP con Herramientas Avanzadas

```python
from fastapi import FastAPI
from fastmcp import MCPServer, Tool, Resource
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()
mcp_server = MCPServer()

# Herramienta para análisis de datos
@mcp_server.tool
def analizar_datos(datos: list, tipo_analisis: str = "estadisticas") -> dict:
    """Analiza una lista de datos numéricos.
    
    Args:
        datos: Lista de valores numéricos
        tipo_analisis: Tipo de análisis a realizar (estadisticas, tendencia)
        
    Returns:
        Resultados del análisis
    """
    df = pd.DataFrame({"valores": datos})
    
    if tipo_analisis == "estadisticas":
        return {
            "media": df["valores"].mean(),
            "mediana": df["valores"].median(),
            "desviacion_estandar": df["valores"].std(),
            "min": df["valores"].min(),
            "max": df["valores"].max()
        }
    elif tipo_analisis == "tendencia":
        return {
            "tendencia": "creciente" if df["valores"].iloc[-1] > df["valores"].iloc[0] else "decreciente",
            "cambio_porcentual": ((df["valores"].iloc[-1] / df["valores"].iloc[0]) - 1) * 100
        }
    else:
        return {"error": "Tipo de análisis no soportado"}

# Herramienta para generar gráficos
@mcp_server.tool
def generar_grafico(datos: list, tipo_grafico: str = "barras", titulo: str = "Gráfico") -> str:
    """Genera un gráfico a partir de datos.
    
    Args:
        datos: Lista de valores numéricos
        tipo_grafico: Tipo de gráfico (barras, linea, pie)
        titulo: Título del gráfico
        
    Returns:
        Imagen del gráfico en formato base64
    """
    plt.figure(figsize=(10, 6))
    
    if tipo_grafico == "barras":
        plt.bar(range(len(datos)), datos)
    elif tipo_grafico == "linea":
        plt.plot(datos)
    elif tipo_grafico == "pie":
        plt.pie(datos, autopct='%1.1f%%')
    else:
        return {"error": "Tipo de gráfico no soportado"}
    
    plt.title(titulo)
    
    # Guardar gráfico en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convertir a base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

# Montar el servidor MCP en FastAPI
app.mount("/mcp", mcp_server.app)

# Ejecutar con: uvicorn nombre_archivo:app --host 0.0.0.0 --port 8000
```

### 4.3 Servidor MCP con Streamable HTTP

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()

# Definición de herramientas disponibles
tools = {
    "suma": {
        "description": "Suma dos números",
        "parameters": {
            "a": {"type": "number", "description": "Primer número"},
            "b": {"type": "number", "description": "Segundo número"}
        }
    },
    "traducir": {
        "description": "Traduce texto a otro idioma",
        "parameters": {
            "texto": {"type": "string", "description": "Texto a traducir"},
            "idioma_destino": {"type": "string", "description": "Idioma destino"}
        }
    }
}

# Implementación de herramientas
async def ejecutar_herramienta(nombre, params):
    if nombre == "suma":
        return {"resultado": params["a"] + params["b"]}
    elif nombre == "traducir":
        # Simulación de traducción
        traducciones = {
            "es": {"hello": "hola", "world": "mundo"},
            "fr": {"hello": "bonjour", "world": "monde"}
        }
        texto = params["texto"].lower()
        idioma = params["idioma_destino"]
        
        if idioma in traducciones and texto in traducciones[idioma]:
            return {"texto_traducido": traducciones[idioma][texto]}
        else:
            return {"texto_traducido": f"[Traducción de '{texto}' a {idioma}]"}
    else:
        return {"error": "Herramienta no encontrada"}

@app.post("/mcp")
async def mcp_streamable(request: Request):
    """Endpoint para comunicación streamable HTTP con MCP."""
    
    # Leer el cuerpo de la solicitud
    body = await request.json()
    
    # Manejar diferentes tipos de mensajes
    if body.get("type") == "init":
        # Inicialización de conexión
        async def stream_response():
            # Enviar mensaje de inicialización
            yield json.dumps({
                "type": "init",
                "status": "ok",
                "server_info": {
                    "name": "Ejemplo MCP Server",
                    "version": "1.0.0",
                    "tools": tools
                }
            }).encode() + b"\n"
        
        return StreamingResponse(stream_response(), media_type="application/json")
    
    elif body.get("type") == "tool_call":
        # Llamada a herramienta
        tool_name = body.get("tool", "")
        params = body.get("params", {})
        
        async def stream_response():
            # Enviar mensaje de inicio
            yield json.dumps({
                "type": "tool_call_start",
                "id": body.get("id", ""),
                "tool": tool_name
            }).encode() + b"\n"
            
            # Simular procesamiento
            await asyncio.sleep(1)
            
            # Ejecutar herramienta
            result = await ejecutar_herramienta(tool_name, params)
            
            # Enviar resultado
            yield json.dumps({
                "type": "tool_call_result",
                "id": body.get("id", ""),
                "result": result
            }).encode() + b"\n"
        
        return StreamingResponse(stream_response(), media_type="application/json")
    
    else:
        # Tipo de mensaje no soportado
        return {"error": "Tipo de mensaje no soportado"}

# Ejecutar con: uvicorn nombre_archivo:app --host 0.0.0.0 --port 8000
```

## 5. Descubrimiento y Registro entre Clientes y Servidores

### 5.1 Descubrimiento Manual

El enfoque más simple es configurar manualmente las URLs de los servidores MCP en los clientes:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

# Configuración manual de servidores MCP
servidores_mcp = [
    MCPTools(transport="streamable-http", url="http://servidor1:8000/mcp", name="servidor1"),
    MCPTools(transport="streamable-http", url="http://servidor2:8000/mcp", name="servidor2"),
    MCPTools(transport="streamable-http", url="http://servidor3:8000/mcp", name="servidor3")
]

# Crear agente con todos los servidores
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=servidores_mcp,
    description="Agente con múltiples servidores MCP",
    markdown=True
)
```

### 5.2 Registro Dinámico con Servicio de Descubrimiento

```python
import requests
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

class MCPDiscoveryService:
    """Servicio para descubrir servidores MCP disponibles."""
    
    def __init__(self, discovery_url):
        self.discovery_url = discovery_url
    
    def discover_servers(self):
        """Descubre servidores MCP disponibles."""
        try:
            response = requests.get(f"{self.discovery_url}/servers")
            if response.status_code == 200:
                return response.json().get("servers", [])
            else:
                print(f"Error al descubrir servidores: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error de conexión: {e}")
            return []
    
    def register_server(self, server_info):
        """Registra un nuevo servidor MCP."""
        try:
            response = requests.post(f"{self.discovery_url}/servers", json=server_info)
            return response.status_code == 200
        except Exception as e:
            print(f"Error al registrar servidor: {e}")
            return False

# Uso del servicio de descubrimiento
discovery_service = MCPDiscoveryService("http://discovery-service:8000/api")

# Descubrir servidores disponibles
available_servers = discovery_service.discover_servers()

# Crear herramientas MCP para cada servidor descubierto
mcp_tools = []
for server in available_servers:
    mcp_tool = MCPTools(
        transport=server.get("transport", "streamable-http"),
        url=server.get("url"),
        name=server.get("name"),
        auth_headers=server.get("auth_headers", {})
    )
    mcp_tools.append(mcp_tool)

# Crear agente con las herramientas descubiertas
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=mcp_tools,
    description="Agente con servidores MCP descubiertos dinámicamente",
    markdown=True
)
```

### 5.3 Implementación de un Servicio de Descubrimiento

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid

app = FastAPI()

# Modelo para información de servidor MCP
class MCPServerInfo(BaseModel):
    name: str
    url: str
    transport: str = "streamable-http"
    description: Optional[str] = None
    capabilities: List[str] = []
    auth_required: bool = False
    auth_type: Optional[str] = None

# Almacenamiento de servidores (en memoria para este ejemplo)
mcp_servers: Dict[str, MCPServerInfo] = {}

@app.post("/api/servers")
async def register_server(server: MCPServerInfo):
    """Registra un nuevo servidor MCP."""
    server_id = str(uuid.uuid4())
    mcp_servers[server_id] = server
    return {"server_id": server_id}

@app.get("/api/servers")
async def get_servers():
    """Obtiene todos los servidores MCP registrados."""
    return {"servers": [{"id": id, **server.dict()} for id, server in mcp_servers.items()]}

@app.get("/api/servers/{server_id}")
async def get_server(server_id: str):
    """Obtiene información de un servidor específico."""
    if server_id not in mcp_servers:
        raise HTTPException(status_code=404, detail="Servidor no encontrado")
    return {"server": mcp_servers[server_id]}

@app.delete("/api/servers/{server_id}")
async def unregister_server(server_id: str):
    """Elimina un servidor del registro."""
    if server_id not in mcp_servers:
        raise HTTPException(status_code=404, detail="Servidor no encontrado")
    del mcp_servers[server_id]
    return {"status": "ok"}

@app.get("/api/servers/capabilities/{capability}")
async def get_servers_by_capability(capability: str):
    """Obtiene servidores que ofrecen una capacidad específica."""
    matching_servers = [
        {"id": id, **server.dict()}
        for id, server in mcp_servers.items()
        if capability in server.capabilities
    ]
    return {"servers": matching_servers}

# Ejecutar con: uvicorn nombre_archivo:app --host 0.0.0.0 --port 8000
```

## 6. Ejecución de Herramientas por LLMs

### 6.1 Proceso de Ejecución de Herramientas

Cuando un LLM necesita ejecutar una herramienta, sigue este proceso:

1. El LLM identifica la necesidad de usar una herramienta
2. El cliente MCP busca la herramienta en los servidores disponibles
3. El cliente envía la solicitud al servidor correspondiente
4. El servidor ejecuta la herramienta y devuelve el resultado
5. El resultado se incorpora al contexto del LLM

### 6.2 Ejemplo de Ejecución de Herramientas con Agno

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
import json

# Configurar cliente MCP
mcp_tools = MCPTools(
    transport="streamable-http",
    url="http://localhost:8000/mcp"
)

# Crear agente
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[mcp_tools],
    description="Agente para demostrar ejecución de herramientas",
    markdown=True
)

# Función para mostrar el proceso de ejecución de herramientas
def ejecutar_con_detalles(prompt):
    print(f"Prompt: {prompt}")
    print("Procesando...")
    
    # Capturar el proceso de razonamiento y ejecución de herramientas
    response = agent.get_response(
        prompt,
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True
    )
    
    print("\nRespuesta final:")
    print(response)

# Ejemplo de ejecución
ejecutar_con_detalles("Suma los números 42 y 58, luego traduce la palabra 'hello' al español")
```

### 6.3 Ejecución de Herramientas con Retroalimentación

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
import asyncio

# Configurar cliente MCP
mcp_tools = MCPTools(
    transport="streamable-http",
    url="http://localhost:8000/mcp"
)

# Crear agente
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[mcp_tools],
    description="Agente con retroalimentación de herramientas",
    markdown=True
)

# Función para manejar retroalimentación de herramientas
async def ejecutar_con_retroalimentacion(prompt):
    print(f"Prompt: {prompt}")
    
    # Iniciar respuesta en streaming
    response_stream = agent.get_response_async(
        prompt,
        stream=True,
        show_full_reasoning=True
    )
    
    # Procesar eventos del stream
    async for event in response_stream:
        if event["type"] == "tool_start":
            # Una herramienta está a punto de ejecutarse
            tool_name = event.get("tool", {}).get("name", "desconocida")
            print(f"Ejecutando herramienta: {tool_name}")
            
        elif event["type"] == "tool_result":
            # Una herramienta ha devuelto un resultado
            tool_name = event.get("tool", {}).get("name", "desconocida")
            result = event.get("result", {})
            print(f"Resultado de {tool_name}: {result}")
            
            # Aquí podrías intervenir o modificar el resultado si es necesario
            
        elif event["type"] == "content":
            # Contenido generado por el LLM
            content = event.get("content", "")
            print(f"LLM: {content}")
    
    # Obtener respuesta final
    final_response = await agent.get_response_async(prompt)
    print("\nRespuesta final:")
    print(final_response)

# Ejecutar ejemplo
asyncio.run(ejecutar_con_retroalimentacion(
    "Analiza los siguientes datos [10, 25, 30, 15, 40], genera un gráfico de barras y explica los resultados"
))
```

### 6.4 Manejo de Errores en Ejecución de Herramientas

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

# Configurar cliente MCP
mcp_tools = MCPTools(
    transport="streamable-http",
    url="http://localhost:8000/mcp"
)

# Crear agente con manejo de errores
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[mcp_tools],
    description="Agente con manejo de errores en herramientas",
    markdown=True,
    tool_error_handling="retry"  # Opciones: "retry", "ignore", "fail"
)

# Función para ejecutar con manejo de errores
def ejecutar_con_manejo_errores(prompt):
    try:
        response = agent.get_response(prompt)
        print(f"Respuesta exitosa: {response}")
        return response
    except Exception as e:
        print(f"Error en la ejecución: {e}")
        
        # Intentar con configuración alternativa
        fallback_agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            description="Agente de respaldo sin herramientas externas",
            markdown=True
        )
        
        fallback_response = fallback_agent.get_response(
            prompt + " (Nota: las herramientas externas no están disponibles)"
        )
        print(f"Respuesta de respaldo: {fallback_response}")
        return fallback_response

# Ejemplo de uso
ejecutar_con_manejo_errores("Suma 25 y 75, luego genera un gráfico de pastel")
```

## 7. Guía para Construir tu Propio Agente

### 7.1 Diseño de la Arquitectura

Para construir tu propio agente con MCP y Agno, sigue estos pasos:

1. **Define los requisitos**:
   - ¿Qué tareas debe realizar el agente?
   - ¿Qué herramientas necesita?
   - ¿Qué LLMs utilizará?

2. **Diseña la arquitectura**:
   - Cliente MCP central
   - Servidores MCP para diferentes dominios
   - Mecanismo de descubrimiento (si es necesario)
   - Interfaz de usuario (si es necesario)

3. **Selecciona los componentes**:
   - LLMs: OpenAI, Anthropic, etc.
   - Framework: Agno
   - Transporte: Streamable HTTP (recomendado)
   - Servidores: FastMCP, implementación personalizada, etc.

### 7.2 Implementación Paso a Paso

#### Paso 1: Configurar el Entorno

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install agno fastapi uvicorn fastmcp
```

#### Paso 2: Implementar Servidores MCP

Crea un archivo `servidor_analisis.py`:

```python
from fastapi import FastAPI
from fastmcp import MCPServer
import pandas as pd
import numpy as np

app = FastAPI()
mcp_server = MCPServer()

@mcp_server.tool
def analisis_estadistico(datos: list) -> dict:
    """Realiza análisis estadístico de una lista de números."""
    series = pd.Series(datos)
    return {
        "media": series.mean(),
        "mediana": series.median(),
        "desviacion_estandar": series.std(),
        "min": series.min(),
        "max": series.max()
    }

@mcp_server.tool
def detectar_anomalias(datos: list, umbral: float = 2.0) -> dict:
    """Detecta valores anómalos en una lista de números."""
    series = pd.Series(datos)
    mean = series.mean()
    std = series.std()
    
    lower_bound = mean - umbral * std
    upper_bound = mean + umbral * std
    
    anomalias = [x for x in datos if x < lower_bound or x > upper_bound]
    indices = [i for i, x in enumerate(datos) if x < lower_bound or x > upper_bound]
    
    return {
        "anomalias": anomalias,
        "indices": indices,
        "total": len(anomalias)
    }

app.mount("/mcp", mcp_server.app)

# Ejecutar con: uvicorn servidor_analisis:app --host 0.0.0.0 --port 8001
```

Crea un archivo `servidor_texto.py`:

```python
from fastapi import FastAPI
from fastmcp import MCPServer

app = FastAPI()
mcp_server = MCPServer()

@mcp_server.tool
def contar_palabras(texto: str) -> dict:
    """Cuenta palabras en un texto."""
    palabras = texto.split()
    return {
        "total_palabras": len(palabras),
        "palabras_unicas": len(set(palabras))
    }

@mcp_server.tool
def resumir_texto(texto: str, max_palabras: int = 50) -> str:
    """Simula resumir un texto (en un caso real usaría un LLM)."""
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    
    # Simulación simple de resumen
    return " ".join(palabras[:max_palabras]) + "..."

app.mount("/mcp", mcp_server.app)

# Ejecutar con: uvicorn servidor_texto:app --host 0.0.0.0 --port 8002
```

#### Paso 3: Implementar el Cliente MCP Principal

Crea un archivo `agente_principal.py`:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
import os

# Configurar API keys (mejor usar variables de entorno)
os.environ["OPENAI_API_KEY"] = "tu-api-key"

# Configurar servidores MCP
servidor_analisis = MCPTools(
    transport="streamable-http",
    url="http://localhost:8001/mcp",
    name="analisis"
)

servidor_texto = MCPTools(
    transport="streamable-http",
    url="http://localhost:8002/mcp",
    name="texto"
)

# Crear agente principal
agente = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[servidor_analisis, servidor_texto],
    description="""
    Eres un asistente analítico que puede procesar datos numéricos y texto.
    Utiliza las herramientas disponibles para proporcionar análisis detallados.
    Explica tus resultados de manera clara y concisa.
    """,
    markdown=True
)

def procesar_consulta(consulta):
    """Procesa una consulta del usuario."""
    print(f"Procesando: {consulta}")
    respuesta = agente.get_response(consulta)
    print("\nRespuesta:")
    print(respuesta)
    return respuesta

# Interfaz simple de línea de comandos
if __name__ == "__main__":
    print("Agente Analítico MCP - Escribe 'salir' para terminar")
    while True:
        consulta = input("\nConsulta: ")
        if consulta.lower() == "salir":
            break
        procesar_consulta(consulta)
```

#### Paso 4: Implementar una Interfaz Web (Opcional)

Crea un archivo `app_web.py`:

```python
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
from agente_principal import procesar_consulta

# Crear directorio para plantillas si no existe
os.makedirs("templates", exist_ok=True)

# Crear plantilla HTML básica
with open("templates/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agente MCP</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; border-radius: 5px; padding: 10px; height: 400px; overflow-y: auto; margin-bottom: 10px; }
            .user-message { background-color: #e6f7ff; padding: 8px; border-radius: 5px; margin-bottom: 10px; }
            .agent-message { background-color: #f0f0f0; padding: 8px; border-radius: 5px; margin-bottom: 10px; }
            .input-container { display: flex; }
            input[type="text"] { flex-grow: 1; padding: 8px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; margin-left: 10px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Agente MCP</h1>
        <div class="chat-container" id="chatContainer">
            <div class="agent-message">Hola, soy tu asistente analítico. ¿En qué puedo ayudarte hoy?</div>
        </div>
        <form action="/consulta" method="post">
            <div class="input-container">
                <input type="text" name="consulta" placeholder="Escribe tu consulta aquí..." required>
                <button type="submit">Enviar</button>
            </div>
        </form>
        
        <script>
            document.querySelector('form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const consulta = document.querySelector('input[name="consulta"]').value;
                const chatContainer = document.getElementById('chatContainer');
                
                // Agregar mensaje del usuario
                chatContainer.innerHTML += `<div class="user-message">${consulta}</div>`;
                document.querySelector('input[name="consulta"]').value = '';
                
                // Agregar mensaje de "pensando..."
                const thinkingDiv = document.createElement('div');
                thinkingDiv.className = 'agent-message';
                thinkingDiv.textContent = 'Pensando...';
                chatContainer.appendChild(thinkingDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                // Enviar consulta al servidor
                const response = await fetch('/consulta', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        'consulta': consulta
                    })
                });
                
                const data = await response.json();
                
                // Reemplazar mensaje de "pensando..." con la respuesta
                thinkingDiv.innerHTML = data.respuesta.replace(/\\n/g, '<br>');
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
        </script>
    </body>
    </html>
    """)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/consulta")
async def handle_consulta(consulta: str = Form(...)):
    respuesta = procesar_consulta(consulta)
    return {"respuesta": respuesta}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Paso 5: Ejecutar el Sistema

1. Inicia los servidores MCP:
   ```bash
   # Terminal 1
   uvicorn servidor_analisis:app --host 0.0.0.0 --port 8001
   
   # Terminal 2
   uvicorn servidor_texto:app --host 0.0.0.0 --port 8002
   ```

2. Inicia el cliente (línea de comandos o web):
   ```bash
   # Para línea de comandos
   python agente_principal.py
   
   # Para interfaz web
   python app_web.py
   ```

3. Accede a la interfaz web en `http://localhost:8000` o interactúa a través de la línea de comandos.

### 7.3 Extensión y Personalización

#### Añadir Nuevos Servidores MCP

Para añadir un nuevo servidor MCP, sigue estos pasos:

1. Implementa el servidor con sus herramientas específicas
2. Configura el transporte (streamable HTTP)
3. Añade el servidor al agente principal:

```python
nuevo_servidor = MCPTools(
    transport="streamable-http",
    url="http://localhost:8003/mcp",
    name="nuevo_servicio"
)

# Añadir a la lista de herramientas del agente
agente.tools.append(nuevo_servidor)
```

#### Cambiar Dinámicamente entre LLMs

```python
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude

# Definir modelos
modelos = {
    "gpt4": OpenAIChat(id="gpt-4o"),
    "claude": Claude(id="claude-3-opus-20240229"),
    "gpt3": OpenAIChat(id="gpt-3.5-turbo")
}

def cambiar_modelo(nombre_modelo):
    """Cambia el modelo del agente."""
    if nombre_modelo in modelos:
        agente.model = modelos[nombre_modelo]
        return f"Modelo cambiado a {nombre_modelo}"
    else:
        return f"Modelo {nombre_modelo} no disponible"
```

## 8. Ejemplos Avanzados y Casos de Uso

### 8.1 Agente de Análisis Financiero

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from agno.tools.reasoning import ReasoningTools

# Servidor MCP para análisis financiero
financial_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8001/mcp"
)

# Crear agente financiero
financial_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        financial_mcp,
        ReasoningTools(add_instructions=True)  # Añadir capacidad de razonamiento
    ],
    description="""
    Eres un analista financiero experto. Tu tarea es analizar datos financieros,
    identificar tendencias, calcular métricas clave y proporcionar recomendaciones
    basadas en datos. Explica tu razonamiento paso a paso.
    """,
    markdown=True
)

# Ejemplo de uso
analysis = financial_agent.get_response("""
Analiza los siguientes datos financieros de una empresa:
- Ingresos Q1: $1.2M
- Ingresos Q2: $1.5M
- Ingresos Q3: $1.3M
- Ingresos Q4: $1.8M
- Gastos Q1: $0.9M
- Gastos Q2: $1.1M
- Gastos Q3: $1.0M
- Gastos Q4: $1.2M

Calcula el margen de beneficio por trimestre, identifica tendencias y proporciona
recomendaciones para mejorar el rendimiento financiero.
""")

print(analysis)
```

### 8.2 Agente de Procesamiento de Documentos

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools
from agno.tools.memory import MemoryTools

# Servidores MCP para procesamiento de documentos
document_processing_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8001/mcp"
)

text_analysis_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8002/mcp"
)

# Crear agente de procesamiento de documentos
document_agent = Agent(
    model=Claude(id="claude-3-opus-20240229"),
    tools=[
        document_processing_mcp,
        text_analysis_mcp,
        MemoryTools()  # Añadir memoria para documentos largos
    ],
    description="""
    Eres un especialista en procesamiento y análisis de documentos. Tu tarea es extraer
    información relevante de documentos, resumir contenido, identificar entidades clave
    y responder preguntas basadas en el contenido del documento.
    """,
    markdown=True
)

# Ejemplo de uso
document_text = """
[Contenido del documento largo aquí...]
"""

# Procesar documento
document_agent.memory.add("documento", document_text)

# Realizar consultas sobre el documento
response = document_agent.get_response("""
Basándote en el documento almacenado en tu memoria:
1. Extrae las 5 entidades principales mencionadas
2. Resume el documento en 3 párrafos
3. Identifica los puntos clave y recomendaciones
""")

print(response)
```

### 8.3 Agente Multi-Modal

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from agno.tools.vision import VisionTools

# Servidores MCP para diferentes modalidades
text_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8001/mcp"
)

image_mcp = MCPTools(
    transport="streamable-http",
    url="http://localhost:8002/mcp"
)

# Crear agente multimodal
multimodal_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        text_mcp,
        image_mcp,
        VisionTools()  # Añadir capacidades de visión
    ],
    description="""
    Eres un asistente multimodal que puede procesar y generar texto e imágenes.
    Puedes analizar imágenes, generar descripciones, crear visualizaciones y
    responder preguntas basadas en contenido visual y textual.
    """,
    markdown=True
)

# Ejemplo de uso con imagen
image_path = "ruta/a/imagen.jpg"

response = multimodal_agent.get_response(
    f"Analiza esta imagen y crea un informe detallado sobre su contenido.",
    images=[image_path]
)

print(response)
```

## 9. Solución de Problemas Comunes

### 9.1 Problemas de Conexión

**Problema**: El cliente MCP no puede conectarse al servidor.

**Soluciones**:
1. Verifica que el servidor esté en ejecución
2. Comprueba la URL y el puerto
3. Asegúrate de que no haya firewalls bloqueando la conexión
4. Verifica que el transporte sea compatible (streamable-http)

```python
# Código para verificar la conexión
import requests

def verificar_servidor_mcp(url):
    try:
        response = requests.post(
            url,
            json={"type": "init"},
            timeout=5
        )
        if response.status_code == 200:
            print(f"Conexión exitosa a {url}")
            print(f"Respuesta: {response.json()}")
            return True
        else:
            print(f"Error de conexión: Código {response.status_code}")
            return False
    except Exception as e:
        print(f"Error de conexión: {e}")
        return False

# Verificar servidores
verificar_servidor_mcp("http://localhost:8001/mcp")
verificar_servidor_mcp("http://localhost:8002/mcp")
```

### 9.2 Problemas con Herramientas

**Problema**: Las herramientas no se ejecutan correctamente.

**Soluciones**:
1. Verifica que la herramienta esté registrada en el servidor
2. Comprueba que los parámetros sean correctos
3. Asegúrate de que el LLM esté formateando correctamente las llamadas

```python
# Código para probar herramientas directamente
import requests
import json

def probar_herramienta(url, herramienta, parametros):
    try:
        response = requests.post(
            url,
            json={
                "type": "tool_call",
                "id": "test_call",
                "tool": herramienta,
                "params": parametros
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"Herramienta {herramienta} ejecutada correctamente")
            print(f"Respuesta: {response.text}")
            return True
        else:
            print(f"Error al ejecutar herramienta: Código {response.status_code}")
            return False
    except Exception as e:
        print(f"Error al ejecutar herramienta: {e}")
        return False

# Probar herramienta
probar_herramienta(
    "http://localhost:8001/mcp",
    "suma",
    {"a": 5, "b": 10}
)
```

### 9.3 Problemas con LLMs

**Problema**: El LLM no utiliza las herramientas disponibles.

**Soluciones**:
1. Asegúrate de que el modelo sea compatible con herramientas (function calling)
2. Verifica que las descripciones de las herramientas sean claras
3. Proporciona instrucciones explícitas al LLM

```python
# Mejorar las instrucciones para el uso de herramientas
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[mcp_tools],
    description="""
    Eres un asistente que debe utilizar las herramientas disponibles para resolver problemas.
    
    IMPORTANTE:
    - Siempre utiliza las herramientas disponibles cuando sea apropiado
    - No intentes simular el resultado de las herramientas
    - Si necesitas realizar cálculos, análisis de datos o procesamiento especializado,
      DEBES usar las herramientas correspondientes
    - Explica tu razonamiento antes de usar cada herramienta
    """,
    markdown=True
)
```

## 10. Recursos Adicionales

### 10.1 Documentación Oficial

- [Documentación de Agno](https://docs.agno.com/)
- [Especificación MCP](https://github.com/mcp-spec/mcp)
- [FastMCP](https://github.com/fastmcp/fastmcp)

### 10.2 Tutoriales y Ejemplos

- [Ejemplos de Agno en GitHub](https://github.com/agno-agi/agno/tree/main/cookbook)
- [Tutoriales de MCP](https://mcp-spec.github.io/tutorials/)

### 10.3 Comunidad y Soporte

- [Foro de Agno](https://community.agno.com/)
- [Canal de Discord de MCP](https://discord.gg/mcp-community)
- [Stack Overflow - Tag: agno](https://stackoverflow.com/questions/tagged/agno)
- [Stack Overflow - Tag: mcp](https://stackoverflow.com/questions/tagged/mcp)

---

Esta documentación te proporciona una guía completa para crear agentes con MCP y Agno, desde conceptos básicos hasta implementaciones avanzadas. Utiliza los ejemplos de código como punto de partida para desarrollar tu propio agente personalizado.

Para mantenerte actualizado, visita regularmente la documentación oficial, ya que MCP y Agno están en constante evolución con nuevas características y mejoras.

¡Buena suerte con tu proyecto de agente!
