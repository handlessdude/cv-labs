import numpy as np

filename_data = [
  {
    'in': 'red-hibiscus.jpg',
    'out': 'red-hibiscus-hist.png',
  },
  {
    'in': 'red.png',
    'out': 'red-hist.png',
  }
]


files_for_color_correction = [
  {
    'subfolder': 'gray_world',
    'in': 'gray_world.png',
    'out': 'gray_world.png',
    'in_hist': 'gray_world_in_hist.png',
    'out_hist': 'gray_world_out_hist.png',
    'additional_args': []
  },
  {
    'subfolder': 'reference_color',
    'in': 'reference_color.png',
    'out': 'reference_color.png',
    'in_hist': 'reference_color_in_hist.png',
    'out_hist': 'reference_color_out_hist.png',
    'additional_args': [
        np.array([1, 1, 255]),  # dst
        np.array([255, 1, 1]),  # src
    ]
  },
]
