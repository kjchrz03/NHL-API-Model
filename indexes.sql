-- ===============================
-- PLAY-BY-PLAY
-- ===============================
CREATE UNIQUE INDEX IF NOT EXISTS pbp_unique_idx
ON all_play_by_play (game_id, sortOrder);

-- ===============================
-- GAME SUMMARY
-- ===============================
CREATE UNIQUE INDEX IF NOT EXISTS gs_unique_idx
ON all_game_summary (game_id);

-- ===============================
-- GOALIE GAMES
-- ===============================
CREATE UNIQUE INDEX IF NOT EXISTS goalie_unique_idx
ON all_goalie_games (game_id, player_id);

-- ===============================
-- TEAM GAME STATS (example)
-- ===============================
CREATE UNIQUE INDEX IF NOT EXISTS team_game_unique_idx
ON all_team_game_stats (game_id, team_id);

-- ===============================
-- STANDINGS SNAPSHOTS (example)
-- ===============================
CREATE UNIQUE INDEX IF NOT EXISTS standings_unique_idx
ON all_standings (season_id, game_date, team_id);
