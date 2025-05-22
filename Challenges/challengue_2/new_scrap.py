import praw 
import json

reddit = praw.Reddit(
    client_id="CoLd6FUVsAop4D9tgPi8dw",
    client_secret="qq2ry8m8_aW8L3UoM49hGwG6-jDQOA",
    user_agent="mcd-programacion-ii"
)

subreddits = [
    "guadalajara",
    "mexico",
    "cdmx",
    "monterrey",
    "queretaro",
    "nayarit",
    "hermosillo",
    "estadodemexico"
]
entries = []
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    print(f"Sub: {sub}")

    comments = subreddit.comments(limit=50000)
    for comment in comments:
        entry = {}
        entry["post_title"] = comment.link_title
        entry["body"] = comment.body
        entry["author"] = str(comment.author)
        entries.append(entry)

json_string = json.dumps(entries)
path = "./comments.json"

print("Writing json... ")
with open(path, mode="w", encoding="utf-8") as json_file:
    json_file.write(json_string)

