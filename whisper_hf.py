import os

# Directories
dirs = {
    'whisper': 'emotyda_data_whisper_aligned',
    'aligned': 'emotyda_data_aligned'
}

# Subfolders in each directory
subfolders = ['train', 'test', 'val']

# Iterate over each subfolder
for subfolder in subfolders:
    # Get the full directory paths
    whisper_dir = os.path.join(dirs['whisper'], subfolder)
    aligned_dir = os.path.join(dirs['aligned'], subfolder)

    # Iterate over every file in the whisper directory
    for filename in os.listdir(whisper_dir):
        # Check if it's a .txt file
        if filename.endswith('.txt'):
            # Get the full file paths
            whisper_file_path = os.path.join(whisper_dir, filename)
            aligned_file_path = os.path.join(aligned_dir, filename)

            # Read whisper file
            with open(whisper_file_path, 'r') as whisper_file:
                whisper_lines = whisper_file.readlines()

            # Read aligned file
            with open(aligned_file_path, 'r') as aligned_file:
                aligned_lines = aligned_file.readlines()

            # New lines for aligned file
            new_aligned_lines = []

            # Iterate over each line in the aligned file
            for aligned_line in aligned_lines:
                # Split line into parts
                aligned_parts = aligned_line.split('|')

                # Find corresponding line in whisper file
                for whisper_line in whisper_lines:
                    whisper_parts = whisper_line.split('|')

                    # If speaker_id and text match, swap tags
                    aligned_parts[2:5] = whisper_parts[2:5]

                # Join parts back together
                new_aligned_line = '|'.join(aligned_parts)

                # Add to new lines
                new_aligned_lines.append(new_aligned_line)

            # Write new lines back to aligned file
            with open(aligned_file_path, 'w') as aligned_file:
                aligned_file.writelines(new_aligned_lines)
