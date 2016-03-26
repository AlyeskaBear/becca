import os 
import worlds.ffmpeg_tools as ft

world_dir = 'becca_world_chase_ball'
stills_directory = os.path.join(world_dir, 'frames') 
movie_filename = 'chase_26.mp4'
ft.make_movie(stills_directory, movie_filename=movie_filename)
