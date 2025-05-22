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
    "EstadoDeMexico"
]
entries = []
subreddits = ["guadalajara"]
for sub in subreddits:
    subreddit = reddit.subreddit(sub)

    top_posts = subreddit.hot(limit=3000)
    
    for post in top_posts:
        entry = {}
        print(f"Title {post.title}" )
        entry["title"] = post.title
        entry["author"] = str(post.author)
        entry["num_comments"] = post.num_comments
        entry["created_utc"] = post.created_utc
        entry["comments"] = []
        submission = reddit.submission(id=post.id)
        comments = submission.comments
        for comment in comments:
            comment_entry = {}
            comment_entry["author"] = str(comment.author)
            comment_entry["body"] = comment.body
            comment_entry["created_utc"] = comment.created_utc
            entry["comments"].append(comment_entry)
        entries.append(entry)

json_string = json.dumps(entries)
path = "./comments.json"

print("Writing json... ")
with open(path, mode="w", encoding="utf-8") as json_file:
    json_file.write(json_string)

