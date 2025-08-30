from openai import OpenAI
import json
import math
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Nastavení API klíčů
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Nahraďte svým OpenAI API klíčem
WEATHER_API_KEY =  os.environ.get("WEATHER_API_KEY")  # Nahraďte svým WeatherAPI klíčem


client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def calculate_area(shape, **kwargs):
    """
    Výpočetní funkce pro různé geometrické tvary
    """
    if shape == "circle":
        radius = kwargs.get("radius", 0)
        return math.pi * radius ** 2
    elif shape == "rectangle":
        width = kwargs.get("width", 0)
        height = kwargs.get("height", 0)
        return width * height
    elif shape == "triangle":
        base = kwargs.get("base", 0)
        height = kwargs.get("height", 0)
        return 0.5 * base * height
    else:
        return None

def get_weather(city):
    """
    Získá aktuální počasí ze WeatherAPI.com
    """
    try:
        # URL pro WeatherAPI
        url = f"http://api.weatherapi.com/v1/current.json"
        
        params = {
            'key': WEATHER_API_KEY,
            'q': city,
            'lang': 'cs'  # Čeština
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()  # Vyhodí výjimku při HTTP chybě
        
        data = response.json()
        
        # Extrakce relevantních informací
        weather_info = {
            "city": data['location']['name'],
            "country": data['location']['country'],
            "temperature": data['current']['temp_c'],
            "condition": data['current']['condition']['text'],
            "humidity": data['current']['humidity'],
            "wind_speed": data['current']['wind_kph'],
            "feels_like": data['current']['feelslike_c']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Chyba při volání Weather API: {str(e)}"}
    except KeyError as e:
        return {"error": f"Neočekávaný formát odpovědi: {str(e)}"}
    except Exception as e:
        return {"error": f"Obecná chyba: {str(e)}"}

# Definice dostupných nástrojů pro LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_area",
            "description": "Vypočítá plochu geometrického tvaru",
            "parameters": {
                "type": "object",
                "properties": {
                    "shape": {
                        "type": "string",
                        "enum": ["circle", "rectangle", "triangle"],
                        "description": "Typ geometrického tvaru"
                    },
                    "radius": {
                        "type": "number",
                        "description": "Poloměr kruhu"
                    },
                    "width": {
                        "type": "number",
                        "description": "Šířka obdélníku"
                    },
                    "height": {
                        "type": "number",
                        "description": "Výška obdélníku nebo trojúhelníku"
                    },
                    "base": {
                        "type": "number",
                        "description": "Základna trojúhelníku"
                    }
                },
                "required": ["shape"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Získá aktuální informace o počasí pro dané město pomocí WeatherAPI.com",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Název města (může být i v češtině)"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def call_llm_with_tools(user_message):
    """
    Hlavní funkce pro volání LLM s nástroji
    """
    try:
        # Počáteční volání LLM
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Jsi užitečný asistent. Můžeš používat nástroje pro výpočty a získávání informací."},
                {"role": "user", "content": user_message}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Kontrola, zda LLM chce použít nástroj
        if message.tool_calls:
            # Příprava zpráv pro další volání
            messages = [
                {"role": "system", "content": "Jsi užitečný asistent. Můžeš používat nástroje pro výpočty a získávání informací."},
                {"role": "user", "content": user_message},
                message
            ]
            
            # Provedení volání nástrojů
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Volám funkci: {function_name}")
                print(f"S argumenty: {function_args}")
                
                # Volání příslušné funkce
                if function_name == "calculate_area":
                    result = calculate_area(**function_args)
                elif function_name == "get_weather":
                    result = get_weather(function_args["city"])
                else:
                    result = "Neznámá funkce"
                
                # Přidání výsledku nástroje do zpráv
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result)
                })
            
            # Druhé volání LLM s výsledky nástrojů
            final_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        else:
            # LLM nechtěl použít nástroj
            return message.content
            
    except Exception as e:
        return f"Chyba při volání API: {str(e)}"

def main():
    """
    Hlavní funkce s příklady použití
    """
    print("=== OpenAI Function Calling Demo s Weather API ===\n")
    
    # Příklady dotazů
    queries = [
        "Jaká je plocha kruhu s poloměrem 5?",
        "Vypočítej plochu obdélníku o rozměrech 10x8",
        "Jaké je počasí v Praze?",
        "Jak je na tom počasí v Londýně?",
        "Kolik je 13+21?"  # Dotaz bez použití nástroje
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Dotaz {i}: {query}")
        response = call_llm_with_tools(query)
        print(f"Odpověď: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    # Kontrola API klíčů před spuštěním
    if WEATHER_API_KEY == "your-weatherapi-key-here":
        print("⚠️  UPOZORNĚNÍ: Nezapomeňte nastavit WEATHER_API_KEY!")
        print("   Registrujte se na https://www.weatherapi.com/ pro zdarma API klíč")
        print()
    
    if OPENAI_API_KEY == "your-openai-api-key-here":
        print("⚠️  UPOZORNĚNÍ: Nezapomeňte nastavit OpenAI API klíč!")
        print()
    
    main()