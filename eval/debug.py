import json
from tqdm import tqdm

with open("results.json") as jp:
    combined_contents = json.load(jp)

score_sum = 0
count = 0
yes_count = 0
no_count = 0
for key, result in tqdm(combined_contents.items()):
    try:
        # Computing score
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        pred = result[0]['pred']
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1
    except:
        print(result)

average_score = score_sum / count
accuracy = yes_count / (yes_count + no_count)
print("Yes count:", yes_count)
print("No count:", no_count)
print("Accuracy:", accuracy)
print("Average score:", average_score)
