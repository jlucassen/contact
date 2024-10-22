import attacker
import defender

def simulate_games(vocab_path, n_games, max_rounds, quiet=False):

    prefix_trie = defender.analyze_vocabulary(vocab_path)
    prefix_data = defender.optimize_conditional_distribution(prefix_trie)
    dist = defender.flatten_distribution(prefix_trie, prefix_data)

    scores = []
    for _ in range(n_games):
        game_dist = dist.copy()
        secret_word = defender.sample(dist)
        revealed = 0
        eliminated = set()
        if not quiet: print(f'-----------SECRET WORD: {secret_word}------------')
        for round in range(1, max_rounds+1):
            game_dist = {k:v for (k,v) in game_dist.items() if k.startswith(secret_word[:revealed])}
            attack = attacker.attack(game_dist, eliminated)
            eliminated.add(attack)
            if not quiet: print(f"Prefix is {secret_word[:revealed]}, attacker guessed {attack}")
            if attack == secret_word:
                if not quiet: print(f"Attacker won by guessing, in round {round}")
                scores.append(round)
                break
            revealed += 1
            if revealed == len(secret_word)-1:
                if not quiet: print(f"Attacker won by revealing, in round {round}")
                scores.append(round)
                break
    print(sum(scores)/len(scores))
        
simulate_games('words_alpha2.txt', 10000, 10000)