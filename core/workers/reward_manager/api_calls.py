import time
import traceback
import json

import requests


def call_api(api_endpoint: str, domain: str, prompt: str, response: str, label: str | list, **kwargs) -> dict:

    timeout = False
    error = ""

    default_output = {
        "score": 0.0,
        "cost": 0.0,
        "reasoning": ""
    }
    # Define timeout in seconds
    timeout_seconds = 90
    start_time = time.time()

    # fix some issues for IF tasks
    if domain == "ifeval":
        # "ground_truth": [{"instruction_id": instruction_id_list, "kwargs": json.dumps(kwargs, ensure_ascii=True)}]
        if isinstance(label, list) and len(label) > 0 and "kwargs" in label[0]:
            try:
                org_kwargs = label[0]["kwargs"]
                if isinstance(org_kwargs, str):
                    for one_label in label:
                        one_label["kwargs"] = json.loads(one_label["kwargs"])
            except Exception as e:
                print(f"Error loading kwargs: {e}")

    try:
        payload = {
            "domain": domain,
            "prompt": prompt,
            "response": response,
            "label": label,
            "timeout": timeout_seconds
        }
        payload.update(kwargs)

        # Call the FastAPI endpoint to execute the code with client-side timeout
        response = requests.post(
            api_endpoint,
            # Server-side timeout (keeping this)
            json=payload,
            timeout=timeout_seconds,  # Client-side timeout
        )
        # Parse the response
        result = response.json()

        # Process the API response
        error = result.get("error") or ""

    except requests.Timeout:
        # Handle client-side timeout specifically
        error = f"Timeout after {timeout_seconds} seconds"
        timeout = True
        result = default_output

    except Exception as e:
        # Capture any other exceptions that occur during the API call
        error_message = f"Error calling API: {str(e)}\n"
        error_traceback = traceback.format_exc()
        error = error_message + error_traceback
        result = default_output

    # Return all captured outputs as a single string
    return {
        "score": result.get("score", 0.0),
        "cost": result.get("cost", 0.0),
        "reasoning": result.get("reasoning", ""),
        "called": True,
        "error": error,
        "timeout": timeout,
        "runtime": time.time() - start_time
    }
