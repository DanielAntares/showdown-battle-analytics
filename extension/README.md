# Showdown Win-Prob Assistant (browser extension)

A live overlay for play.pokemonshowdown.com driven by this repo's models: win
probability, the ranked advisor (Fast 1-turn / Deep ~5-turn), the opponent's
likely next action, and a ready `/choose` command. Works in **your own** battles
— public or private — because it reads the battle stream inside your logged-in
client. (Spectating someone else's private game remains impossible.)

## Setup

1. Start the local bridge (loads the models once):

```
run_assistant.bat
```

2. Load the extension (Chrome/Edge):
   - open `chrome://extensions`
   - enable **Developer mode**
   - **Load unpacked** -> select this `extension/` folder

3. Open https://play.pokemonshowdown.com and start a battle. The panel appears
   bottom-right and updates every time the game asks you for a decision.

## Modes

- **Manual (default):** the panel shows the recommendation; you click the move.
- **Auto-play:** the extension sends the recommended `/choose` itself after a
  1–3 s randomized delay. Toggle it in the panel.

⚠️ **Auto-play on the ranked ladder is botting under Pokémon Showdown's rules
and can get your account locked or banned.** The toggle defaults to off. Using
it in challenges/friendlies with consenting opponents is the safer use.

## Notes

- Chrome/Edge (Manifest V3, needs `world: MAIN` content scripts — Chrome 111+).
- The bridge must be running; if the panel says "bridge offline", start
  `run_assistant.bat`.
- Nothing leaves your machine: the extension talks only to `127.0.0.1:8765`.
