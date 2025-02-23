def test_rust_game_creation():
    try:
        from catanatron_rust import Game
        game = Game(4)
        assert game.get_num_players() == 4
        print("\nStarting game simulation...")
        game.play()
    except ImportError as e:
        print(f"Failed to import catanatron_rust: {e}")
        raise 