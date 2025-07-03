import asyncio
import datetime
import json
import os
import re
import dotenv
import pathlib
import requests
from openai import AsyncOpenAI
import numpy as np


from reasoning_prompts import (
    FERMI_METHOD_PROMPT, 
    NAIVE_DIALECTIC_PROMPT, 
    PROPOSE_EVALUATE_SELECT_PROMPT, 
    BAYESIAN_REASONING_PROMPT, 
    ANTI_BIAS_PROMPT, 
    TIPPING_PROMPT, 
    SIMULATED_DIALOGUE_PROMPT, 
    BACKWARD_REASONING_PROMPT,
    METACULUS_BINARY_PROMPT
)

dotenv.load_dotenv()
# METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

MODEL_NAME = "o3"
FILE = "data/q4-2024_binary_resolved_metaculus_questions.json"


BINARY_REASONING_PROMPTS = [
    ("FERMI_METHOD", FERMI_METHOD_PROMPT),
    ("NAIVE_DIALECTIC", NAIVE_DIALECTIC_PROMPT),
    ("PROPOSE_EVALUATE_SELECT", PROPOSE_EVALUATE_SELECT_PROMPT),
    ("BAYESIAN_REASONING", BAYESIAN_REASONING_PROMPT),
    ("ANTI_BIAS", ANTI_BIAS_PROMPT),
    ("TIPPING", TIPPING_PROMPT),
    ("SIMULATED_DIALOGUE", SIMULATED_DIALOGUE_PROMPT),
    ("BACKWARD_REASONING", BACKWARD_REASONING_PROMPT),
    ("METACULUS_BINARY", METACULUS_BINARY_PROMPT)
]
BINARY_PROMPT_TEMPLATE = """
Question
{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}

Your research assistant says:
{summary_report}

Today is {today}.

{reasoning_prompt}

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

#Original Metaculus Binary prompt beginning
# 
# # BINARY_PROMPT_TEMPLATE = """
# You are a professional forecaster interviewing for a job.

# Your interview question is:
# {title}

# Question background:
# {background}

# This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
# {resolution_criteria}

# {fine_print}

# {reasoning_prompt}

# The last thing you write is your final answer as: "Probability: ZZ%", 0-100
# """

client = AsyncOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        max_retries=2,
    )

CONCURRENT_REQUESTS_LIMIT = 9
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


def iso_to_mmddyyyy(iso_str: str) -> str:
    """
    Convert ISO-8601 strings like '2025-06-09T16:00:00Z'
    to '06/09/2025'.  Assumes the Z suffix (UTC).
    """
    if iso_str is None:
        return datetime.datetime.now().strftime("%m/%d/%Y")
    dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%m/%d/%Y")

def run_research(question: str, end_date_search: str) -> str:
    research = ""
    if PERPLEXITY_API_KEY:
        research = call_perplexity(question, end_date_search)
    else:
        research = "No research done"

    # print(f"########################\nResearch Found:\n{research}\n########################")
    return research

def call_perplexity(question: str, end_date_search: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    api_key = PERPLEXITY_API_KEY
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",  # this is a system prompt designed to guide the perplexity assistant
                "content": """
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.
                """,
            },
            {
                "role": "user",  # this is the actual prompt we ask the perplexity assistant to answer
                "content": question,
            },
        ],
        "search_before_date_filter": end_date_search,
        "last_updated_before_filter": end_date_search,
    }
    response = requests.post(url=url, json=payload, headers=headers)
    if not response.ok:
        raise Exception(response.text)
    content = response.json()["choices"][0]["message"]["content"]
    return content


async def call_llm(prompt: str, model: str = MODEL_NAME, temperature: float = 0.3) -> str:
    """
    Makes a streaming completion request to OpenAI's API with concurrent request limiting.
    """
    async with llm_rate_limiter:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("No answer returned from LLM")
        return answer
    
def extract_percentage_and_convert_to_decimal_from_response(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)\s?%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number/ 100.0 # Convert to decimal
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

async def run_reasoning_method(question_details: dict, reasoning_name: str, reasoning_prompt: str):

    filled_prompt = BINARY_PROMPT_TEMPLATE.format(
        title=question_details["title"],
        today=question_details["today"],
        background=question_details["description"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        summary_report=question_details["summary_report"],
        reasoning_prompt=reasoning_prompt
    )
    response = await call_llm(filled_prompt)
    probability = extract_percentage_and_convert_to_decimal_from_response(response)
    return reasoning_name, probability, response, filled_prompt

async def process_binary_question(question_details: dict):
    resolution = question_details["resolution"].lower().strip()
    ground_truth = 1 if resolution == "yes" else 0

    end_date_used = iso_to_mmddyyyy(question_details["open_time"])
    question_details["summary_report"] = run_research(question_details["title"], end_date_used)
    question_details["today"] = end_date_used

    results = await asyncio.gather(
        *[run_reasoning_method(question_details, name, prompt) for name, prompt in BINARY_REASONING_PROMPTS]
    )
    individual_forecasts = {}
    individual_briers = {}
    responses = {}
    filled_prompts = {}

    for name, prob, response, filled_prompt in results:
        individual_forecasts[name] = prob
        individual_briers[name] = (prob - ground_truth) ** 2
        responses[name] = response
        filled_prompts[name] = filled_prompt
    
    ensemble_forecast = np.mean(list(individual_forecasts.values()))
    ensemble_brier = (ensemble_forecast - ground_truth)**2
    
    return {
        "question_id": question_details["id"],
        "title": question_details["title"],
        "resolution": question_details["resolution"],
        "ground_truth": ground_truth,
        "individual_forecasts": individual_forecasts,
        "individual_brier_scores": individual_briers,
        "ensemble_forecast": ensemble_forecast,
        "ensemble_brier": ensemble_brier,
        "summary_report": question_details["summary_report"],
        "research_end_prompt_date": question_details["today"],
        "filled_prompts": filled_prompts,
        "responses": responses
    }


async def binary_main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    questions = [
        q for q in questions
        if q["resolution"].lower().strip() in ["yes", "no"] #to remove ambigious resolved questions
    ]
    # questions = questions[:20]
    all_results = []
    for i, question_details in enumerate(questions):
        print(f"Processing Question {i+1} / {len(questions)}: {question_details['title']}")
        result = await process_binary_question(question_details)
        all_results.append(result)

        # Save partial progress after each question (safety)
        with open(output_dir/f"binary_experiment_results_{timestamp}.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nFinished Question {i+1}: '{result['title']}'")
        for name, _ in BINARY_REASONING_PROMPTS:
            brier = result["individual_brier_scores"][name]
            print(f" - {name}: Brier = {brier:.4f}")
        print(f"Ensemble Brier: {result['ensemble_brier']:.4f}")
        print("-" * 40)

    brier_sums = {name: 0.0 for name, _ in BINARY_REASONING_PROMPTS}
    ensemble_brier_sum = 0.0
    N = len(all_results)

    for result in all_results:
        for name, _ in BINARY_REASONING_PROMPTS:
            brier_sums[name] += result["individual_brier_scores"][name]
        ensemble_brier_sum += result["ensemble_brier"]

    mean_brier_scores = {name: brier_sums[name] / N for name in brier_sums}
    mean_ensemble_brier = ensemble_brier_sum / N

    sorted_scores = sorted(mean_brier_scores.items(), key=lambda x: x[1])
    ranked_brier_scores = [{"name": name, "score": score} for name, score in sorted_scores]

    final_report = {
        "prompt_template_used": BINARY_PROMPT_TEMPLATE,
        "results": all_results,
        "summary":{
            "model": MODEL_NAME,
            "ranked_mean_brier_scores": ranked_brier_scores,
            "ensemble_mean_brier": mean_ensemble_brier
        }
    }

    with open(output_dir / f"binary_experiment_results_{timestamp}.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("Saved experiment results as json.")


    print("\nMean Brier Scores (ranked):")
    for rank, (name, score) in enumerate(sorted_scores, 1):
        print(f"{rank}. {name}: {score:.4f}")

    print(f"\nEnsemble Mean Brier: {mean_ensemble_brier:.4f}")

if __name__ == "__main__":
    asyncio.run(binary_main())
