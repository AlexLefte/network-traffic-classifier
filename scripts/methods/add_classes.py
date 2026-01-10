import pandas as pd

# Define the category mappings
CHAT = [0, 1, 21, 2, 3, 4]  # aim, facebook_chat, gmail, hangouts_chat, skype_chat, icq
EMAIL = [5]  # email
FILE_TRANSFER = [6, 7, 8, 9]  # ftps, scp, sftp, skype_file
STREAMING = [10, 11, 12, 13]  # netflix, spotify, vimeo, youtube
VOIP = [14, 15, 16, 17]  # facebook_audio, hangouts_audio, skype_audio, voipbuster
VIDEO_CALL = [18, 19, 20]  # facebook_video, skype_video, hangouts_video

# Read your original CSV
df = pd.read_csv('features_timeout_15s_2pkts_each_dir.csv')  # Replace with your actual CSV filename

# Create a new column with the category name
def map_label_to_category(label):
    if label in CHAT:
        return 'chat'
    elif label in EMAIL:
        return 'email'
    elif label in FILE_TRANSFER:
        return 'file'
    elif label in STREAMING:
        return 'streaming'
    elif label in VOIP:
        return 'voip'
    elif label in VIDEO_CALL:
        return 'video_call'
    else:
        return 'unknown'

# Apply the mapping
df['Category'] = df['Label'].apply(map_label_to_category)

# Save to new CSV
df.to_csv('features_timeout_15s_2pkts_each_dir_5_class.csv', index=False)

# Print summary
print("Category distribution:")
print(df['Category'].value_counts())
print(f"\nTotal rows: {len(df)}")