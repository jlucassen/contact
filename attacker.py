def attack(flat_dist, eliminated):
    filtered = {k:v for (k,v) in flat_dist.items() if k not in eliminated}
    return sorted(filtered, key=lambda x: x[1], reverse=True)[0]