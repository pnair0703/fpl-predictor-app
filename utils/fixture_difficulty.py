# utils/fixture_difficulty.py

# Official FDR ranges are 1 to 5.
# We keep a simple baseline mapping here.
TEAM_FDR = {
    1: 3,   # Arsenal
    2: 3,   # Aston Villa
    3: 4,   # Bournemouth
    4: 4,   # Brentford
    5: 2,   # Brighton
    6: 2,   # Burnley
    7: 4,   # Chelsea
    8: 3,   # Crystal Palace
    9: 2,   # Everton
    10: 5,  # Fulham
    11: 1,  # Liverpool
    12: 1,  # Luton
    13: 1,  # Man City
    14: 2,  # Man United
    15: 3,  # Newcastle
    16: 4,  # Nottingham Forest
    17: 4,  # Sheffield Utd
    18: 2,  # Spurs
    19: 5,  # West Ham
    20: 4,  # Wolves
}

# We will refine this later using live fixture API data.
