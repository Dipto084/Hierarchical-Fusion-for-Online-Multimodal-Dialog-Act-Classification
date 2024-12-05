import os

# Path to the folder
folder_path = "emotyda_data_whisper_aligned"

# Dictionary to store the tag3 counts and details
tag3_counts = {}

# DA tag names
tag_names = {
    'g': 'Greeting',
    'q': 'Question',
    'ans': 'Answer',
    'o': 'Statement-Opinion',
    's': 'Statement-NonOpinion',
    'ap': 'Apology',
    'c': 'Command',
    'ag': 'Agreement',
    'dag': 'Disagreement',
    'a': 'Acknowledge',
    'b': 'Backchannel',
    'oth': 'Others'
}

# Function to count the tag3 occurrences
def count_tag3(file_path):
    with open(file_path, "r") as file:
        for line in file:
            _, _, _, _, tag3, _ = line.strip().split("|")
            if tag3 not in tag3_counts:
                tag3_counts[tag3] = [tag_names.get(tag3, ''), 0, 0]
            tag3_counts[tag3][1] += 1

# Process each folder
sub_folders = ["train", "test", "val"]

for sub_folder in sub_folders:
    sub_folder_path = os.path.join(folder_path, sub_folder)

    # Process each file in the subfolder
    for file_name in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, file_name)
        count_tag3(file_path)

# Calculate the total count across all folders
total_count = sum(count[1] for count in tag3_counts.values())

# Sort the tag3 counts in descending order
sorted_tag3_counts = sorted(tag3_counts.items(), key=lambda x: x[1][1], reverse=True)

# Calculate the percentage for each tag and cumulative percentage
cumulative_percentage = 0
for tag, details in sorted_tag3_counts:
    tag_count = details[1]
    percentage = (tag_count / total_count) * 100
    details[2] = percentage
    cumulative_percentage += percentage
    details.append(cumulative_percentage)

# Print the tag details in descending order with cumulative percentage
for tag, details in sorted_tag3_counts:
    print(f"Tag3: {tag}")
    print(f"DA Tag: {details[0]}")
    print(f"Total Count: {details[1]}")
    print(f"Percentage: {details[2]:.2f}%")
    print(f"Cumulative Percentage: {details[3]:.2f}%")
    print()
