erged_df = pd.merge(first_merged_df, team, on='TEAM', how='left')
third_merged_df = pd.merge(second_merged_df, matchups, on=['TEAM', 'YEAR'], how='right')

fourth_merged_df = pd.merge(third_merged_df, _538_ratings, on=['TEAM', 'YEAR'], how='left', suffixes=('_kenpom', '_538'))
fifth_merged_df = pd.merge(fourth_merged_df, Bartowick_Away_Neutral, on=['TEAM', 'YEAR'], how='left', suffixes=('_538', '_away_neutral'))
sixth_merged_df = pd.merge(fifth_merged_df, Bartowick_Away, on=['TEAM', 'YEAR'], how='left', suffixes=('_away_neutral', '_away'))
seventh_merged_df = pd.merge(sixth_merged_df, Bartowick_Home, on=['TEAM', 'YEAR'], how='left', suffixes=('_away', '_home'))
eighth_merged_df = pd.merge(seventh_merged_df, Bartowick_Neutral, on=['TEAM', 'YEAR'], how='left', suffixes=('_home', '_neutral'))
ninth_merged_df = eighth_merged_df
tenth_merged_df = eighth_merged_df
eleventh_merged_df = eighth_merged_df 
twelfth_merged_df = eighth_merged_df
thirteenth_merged_df = eighth_merged_df
fourteenth_merged_df = eighth_merged_df
fifteenth_merged_df = eighth_merged_df
sixteenth_merged_df = pd.merge(fifteenth_merged_df, Heat_Check_Tournament_Indes, on=['TEAM', 'YEAR'], how='left')
seventeenth_merged_df = pd.merge(sixteenth_merged_df, Public_Picks, on=['TEAM', 'YEAR'], how='left')
eighteenth_merged_df = pd.merge(seventeenth_merged_df, Resumes, on=['TEAM', 'YEAR'], how='left')
nineteenth_merged_df = pd.merge(eighteenth_merged_df, Seed_Results, on=['TEAM', 'YEAR'], how='left')
twentieth_merged_df = pd.merge(nineteenth_merged_df, Shooting_Splits, on=['TEAM', 'YEAR'], how='left')
twentyfirst_merged_df = pd.merge(twentieth_merged_df, Tournament_Locations, on=['TEAM', 'YEAR'], how='left')
twentysecond_merged_df = pd.merge(twentyfirst_merged_df, Tournament_Simulation, on=['TEAM', 'YEAR'], how='left')
final_merged_df = pd.merge(twentysecond_merged_df, Upset_Count, on=['TEAM', 'YEAR'], how='left')
final_merged_df = pd.merge(final_merged_df, Upset_Seed_Info, on=['TEAM', 'YEAR'], how='left')