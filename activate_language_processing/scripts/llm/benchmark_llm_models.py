from ollama import chat

import json
import timeit

MODELS = ["nlp-gemma4", "nlp-gemma4-e4b", "nlp-qwen"]


def test_user_input(u_input: list, args):
    results = []
    for model in MODELS:
        for user_input in u_input:
            print(f"MODEL: {model}\n")
            print(user_input)
            t1 = timeit.default_timer()
            response = chat(
                model=model,
                messages=[{"role": "user", "content": user_input}],
                think=args.think,
                format="json",
            )
            t2 = timeit.default_timer()
            inference_time = t2 - t1
            output = response.message.content.strip()
            print(f"RESPONSE: {output}\n")

            # Check for valid JSON structure in response
            try:
                parsed_response = json.loads(output)
                is_valid_json = True
            except json.JSONDecodeError:
                parsed_response = None
                is_valid_json = False

            results.append({
                "model": model,
                "user_input": user_input,
                "response_raw": output,
                "response_parsed": parsed_response,
                "is_valid_json": is_valid_json,
                "response_time_seconds": round(inference_time, 2),
            })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="Path to test data")
    parser.add_argument("-s", "--save", required=False, help="Where to save the results")
    parser.add_argument("--think", action="store_true", required=False, help="Enable or disable think mode")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        user_inputs = [line.strip() for line in f.readlines()]

    results = test_user_input(user_inputs, args)

    save_file = args.save if args.save else "llm_test_results.json"

    with open(save_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {save_file}")


if __name__ == "__main__":
    main()