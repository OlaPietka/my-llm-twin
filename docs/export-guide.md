# How to Export Your DMs

You need to export your Messenger messages as JSON files.
This is a one-time manual step — Meta doesn't offer an API for this.

## Messenger

1. Go to **facebook.com**
2. Click your **profile icon** (top right)
3. Click **Settings and privacy**
4. Click **Settings**
5. This opens **Account Center**
6. Go to **Your information and permissions**
7. Click **Download your information**
8. Choose **Export to device**
9. Select **Messages** only
10. Pick **JSON** format
11. Select your time range
12. Click **Start export**

You'll get an email when the export is ready (can take minutes to hours).
Download the ZIP and unzip it into `data/raw/`.

## After exporting

Unzip into `data/raw/`:

```
data/raw/
  your_facebook_activity/
    messages/
      inbox/
        ...
```

Then run `my-llm-twin parse` to process the messages.

## Note on Instagram

Instagram DM exports are not supported in v1. Instagram DMs tend to be
mostly reels, reactions, and media shares with very little actual text
content — not useful for training a text-based model.
