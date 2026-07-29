All development is done on pwa-5090 ~/code/cpp/chess/nnue
read AGENTS.md there and the associated SKILL.md's

The best net is currently ~ -154 elo from SF and ~ -122 from Berserk:

enyo-1.30.0-rc3.nn vs berserk-9b84c340af7e.nn (1500 games):
elo=-122.2  llr=-2.29/2.20 (-104%)  los=0.0%  ci=31.9  draw=35.4%

enyo-1.30.0-rc3.nn vs nn-0ee0657fb25e.nnue (1500 games):
elo=-154.4  llr=-22.66/690.78 (-3%)  los=0.0%  ci=15.4  draw=32.9%


Update: the currently best net is always ~/assts/nets/reference.net (enyo-1.32.0-rc10.nn)

The NNUE architecture is failry close to Berserk, but I have been able to close the gap further.
I have tried different architectures, selfplay, stockfish binpacks, and additional features such
as FullThreats.

Look at the git log to investigate the different lineages.

I am contemplating re-doing an earlier experiment and have a competition between multiple architecutes and/or features, trained without using existing tensors.
