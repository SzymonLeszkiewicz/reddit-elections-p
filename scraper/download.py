import os
import pandas as pd
import praw
from dotenv import load_dotenv
from qwlist.qwlist import Lazy, QList
from prawcore.exceptions import Forbidden, TooManyRequests
from time import sleep

load_dotenv()

def is_comment(comment) -> bool:
    return all(
        hasattr(comment, attr)
        for attr in ('body', 'author', 'id', 'created_utc', 'score')
    )

reddit = praw.Reddit(
    client_id=os.environ['REDDIT_CLIENT_ID'],
    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
    user_agent=os.environ.get('REDDIT_USER_AGENT', 'reddit-political-discourse/1.0'),
    check_for_async=False
)
# Loading the subreddits
with open('./subreddits.txt', 'r', encoding='utf-8') as file:
    subreddits = (
        QList(file.readlines())
        .map(str.strip)
        .map(reddit.subreddit)
        .collect()
    )
subreddits = {sub.display_name.lower(): sub for sub in subreddits}


# Load the usernames of users to download
with open('users.txt', 'r', encoding='utf-8') as file:
    usernames = set([line.strip() for line in file.readlines()])


# Excluding the already downloaded users
OUTPUT = '../data/comments.csv'

already_downloaded = set()
if os.path.exists(OUTPUT):
    print(f'{OUTPUT} exists. Loading previously downloaded users...')
    already_downloaded = set(pd.read_csv(OUTPUT, encoding='utf-8', sep=';')['user_name'])

weird_users = set()

# Download loop
users_to_download = usernames - already_downloaded

while len(users_to_download) > 0:
    print(f'Users to download: {len(users_to_download)}')

    try:
        headers = ''
        if not os.path.exists('../data/comments.csv'):
            headers = 'subreddit_id;subreddit_name;user_id;user_name;comment_id;comment_timestamp;text;score;replies_count\n'

        for user in list(users_to_download):
            redditor = reddit.redditor(user)
            comments_rcount = (
                Lazy(redditor.comments.new(limit=None))
                .filter(is_comment)
                .filter(lambda comment: (
                    comment
                    .subreddit
                    .display_name
                    .lower() in subreddits
                ))
                .map(lambda comment: (comment, len(comment.replies)))
            )
            with open('../data/comments.csv', 'a+', encoding='utf-8') as file:
                file.write(headers)
                for comment, replies_count in comments_rcount:
                    body = comment.body.replace('\n', ' ').replace('\r', '').replace('"', "'")
                    line = f'{comment.subreddit.id};"{comment.subreddit.display_name}";{redditor.id};"{redditor.name}";{comment.id};{comment.created_utc};"{body}";{comment.score};{replies_count}\n'
                    file.write(line)
                    if redditor.name in users_to_download:
                        users_to_download.remove(redditor.name)
    except TooManyRequests as e:
        print('Rate limit exceeded. Waiting for a while...')
        sleep(60.0)
        already_downloaded = set(pd.read_csv('../data/comments.csv', encoding='utf-8', sep=';')['user_name'])
        users_to_download = users_to_download - already_downloaded
    except Forbidden as e:
        print('Forbidden. (Maybe private account, I dunno...)')
        users_to_download.remove(user)
    except Exception as e:
        print(f'{e}')
        print('I don\'t know what to do so I continue...')
        already_downloaded = set(pd.read_csv('../data/comments.csv', encoding='utf-8', sep=';')['user_name'])
        users_to_download = users_to_download - already_downloaded
        sleep(60.0)


print('DONE')