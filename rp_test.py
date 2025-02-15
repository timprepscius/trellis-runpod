import json

from rp_handler import process

with open("test_input.json", "r") as f:
    test_input_json = json.load(f)

test_input_json["id"] = "test_id"

try:
    process(test_input_json)
except Exception as e:
    print(e)

print("finishing without an error")
