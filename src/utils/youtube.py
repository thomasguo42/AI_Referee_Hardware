from pytubefix import YouTube
import sys

def download_youtube_video(url, output_path="."):
    try:
        # Create YouTube object
        yt = YouTube(url)
        
        # Get the highest resolution MP4 video stream with audio
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if video is None:
            print("No suitable video stream found. Try a different video or check restrictions.")
            return
        
        # Print selected resolution for confirmation
        print(f"Downloading: {yt.title}")
        print(f"Selected resolution: {video.resolution}")
        video.download(output_path)
        print("Download completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Suggestions:")
        print("- Ensure the URL is valid and the video is publicly accessible.")
        print("- Update pytubefix: `pip install --upgrade pytubefix`")
        print("- Test with a different video URL.")
        sys.exit(1)

# Example usage
video_url = "https://www.youtube.com/watch?v=A74YChOkSpY&list=PLJpCdaWK6PVo7yhl9Hl9oTmoa0vz191Ev"
download_youtube_video(video_url)