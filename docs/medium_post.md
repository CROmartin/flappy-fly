# I put a fruit fly's brain in a Flappy Bird clone

*The headline is the stunt. The interesting part is why a scanned brain does not play anything until you borrow the maps from experiments that already poked it.*

I wired the adult male fruit fly connectome into a Flappy-style game. One hundred sixty-six thousand neurons. About twenty-five million connections. A frozen copy of a real nervous system sits between the pipes and the flap button. When the looming signal gets loud enough, the fly jumps the gap.

That is the post people share. Here is the part that makes the stunt mean something.

A 1:1 scan of a brain does not arrive as a working pilot. It arrives as a wiring diagram. Nobody labeled neuron 84321 with "this one sees a pipe" or "this one means jump." The scan tells you who talks to whom. It does not tell you what any of them are for. What makes a connectome usable is an older, slower literature: labs that showed the animal a picture, touched a bristle, or waved an odor, and wrote down which cells answered.

## The file is a map of wires, not a mind

The graph comes from the MaleCNS electron-microscopy reconstruction of an adult male *Drosophila* central nervous system: 166,700 neurons and 25,582,938 connections, released by FlyEM at Janelia with Cambridge, the MRC Laboratory of Molecular Biology, and Google Research. A small simulator (`flybrain`) turns synapse counts into signed connection strengths and steps a simplified spiking model forward in time.

That is already several steps away from the animal. The weights are normalized counts of synapses, not measured physiological strengths. The cells are a leaky integrate-and-fire approximation with hand-set dynamics and noise. There are no eyes, no muscles, no wings, and no synaptic learning. The biological fly never saw a pipe. "CONNECTOME ONLINE" on the dashboard is a joke sitting on top of a real adjacency list.

If you load that graph and press play, nothing in it knows the rules of the game. Pipes are not a stimulus the fly evolved to parse. Pixels are not spikes. A descending neuron does not come with a Flappy Bird binding. Until something injects current into cells that already have a known job, the network is a silent circuit board.

## Usable means: we already know what wakes it up

Fly neuroscience spent decades doing the unglamorous version of this project. Stimulate, record, name the cell type, stimulate again. After enough of that, a wiring diagram becomes addressable. You can point at a population and say what kind of event in the world tends to drive it.

Three of those worlds matter here.

**Vision.** The compound eye feeds a stack of optic-lobe neurons. Some of them are specialists. LC4 and LPLC2 respond to looming: an object expanding fast, the visual signature of something about to hit you, which in the fly often triggers an escape takeoff. LC10a tracks small objects. LPLC1 cares about looming and motion too. These are not "Flappy Bird neurons." They are cells with a published relationship to a visual event.

**Touch, wind, and vibration.** The body is covered in mechanosensory bristles. The antenna (Johnston's organ) hears and feels air moving past the head. A puff on a leg and a gust on the antenna land in different, named populations. If I had wanted the game to speak "wind," I would have looked up those cells and injected there. The scan would not have suggested it.

**Smell.** Olfactory receptor neurons on the antenna and maxillary palp report odors into the antennal lobe, and from there into the mushroom body and the lateral horn. That is the third sense people forget when they picture a fly as a tiny camera with wings. Present banana or a predator odor and a different set of cells wakes up. Taste, temperature, and humidity have their own doorways too. The principle is the same for all of them: the meaning lives in the stimulus experiment, and the connectome lets you follow the wires afterward.

I only used the visual doorway. Touch and smell are in the post because they are the reason the method generalizes. A connectome without those maps is a phone book. A connectome with them is a switchboard you can call.

## What the game actually whispers into the visual cells

Each frame, a hand-written encoder turns the game into six numbers between 0 and 1, and those numbers are injected as current into real cell types:

| Channel | What the game stuffs in | Why that cell type was a candidate |
|---|---|---|
| LC4 | "Flapping now looks like it would hit the ceiling" | Looming / collision visual neurons |
| LPLC2 | "Waiting looks like it would hit the floor" | Looming visual neurons |
| LC10a, left | The fly is below the gap | Object-tracking visual neurons |
| LC10a, right | The fly is above the gap | Same population, other side |
| LPLC1, left | The fly is falling | Motion-sensitive visual neurons |
| LPLC1, right | The fly is rising | Same population, other side |

Left-means-down and right-means-up is my convention, not a result from the fly. So is the whole translation from "gap in a pipe" into "looming." I am renting the cells' known enthusiasm for expanding, dangerous-looking visual events, and I am telling them a story about pipes in that accent.

The encoder is where the intelligence of the demo mostly sits. It knows the fly's radius, the pipe lips, gravity, and a short lookahead: if I flap, do I hit the top; if I wait, do I hit the bottom. That calculation is ordinary game code. The connectome never sees coordinates. It sees six fake stimuli, dressed up as the kind of visual drive those neurons were characterized with.

## Getting a flap back out

On the way out, the same rule applies. I do not read "the brain." I read the descending neurons: 1,314 cells whose axons leave the brain toward the ventral nerve cord, the wires that normally command the body.

One of them is famous. DNp01 is a giant descending neuron in the visual escape pathway. Looming in LC4 and LPLC2 can drive it, and in the fly that drive is part of the takeoff reflex. Instinct mode in the game is almost embarrassingly direct: if that cell's recent activity crosses a threshold, flap. A trace threshold is not a probability, and it is not the fly deciding. It is a known output line, thresholded.

Brain mode is one step less cute. The connectome stays frozen. Nothing in the 25 million connections is trained. A logistic readout looks at the activity of all 1,314 descending neurons, compresses them, and learns FLAP versus WAIT from a scripted teacher that already knows how to clear the pipes. The classifier learns which activity patterns matched the teacher's button presses. The fly does not learn the game.

There is also a small actuator rule on top of the classifier, tuned by hand: when the position channels say the fly is too high, dampen the urge to flap and let gravity do the work; when it is too low, boost the flap. That rule is not in the connectome either. It is an admission that a raw readout loves to flap too often.

## The part the screenshot leaves out

The first honest runs were a bad advertisement and a good result. With a naive encoding, the trained fly readout cleared almost nothing: on 30 unseen seeds it averaged 0.07 pipes, worse than flapping at random. A logistic regression on three hand-built numbers, skipping the connectome entirely, also crashed. High recall and low precision meant the policy flapped early, flew into a state it had never trained on, and died. A brain scan in the loop was not a shortcut around control theory.

Later versions rewrote the sensory story. Upper threat and lower threat stopped being the same "something is close" signal. A short lookahead asked what flapping would do versus what waiting would do. Position was strengthened. The too-high / too-low bias was added so the policy could choose to fall. After that, on the same 30 seeds and a 15-second cap, instinct, the descending-neuron readout, and the three-input baseline all survived every episode.

Read that again before you put it on a slide. The moment the demo looks like magic is the moment the interface got specific: the right visual cell types, a game-shaped meaning assigned by me, a frozen circuit, a small trained output layer, and a hand-written bias that tells it to stop jumping when it is already too high. The wiring diagram did real work in the loop. It did not supply the strategy.

## So what is the stunt, once you say it plainly?

A real connectome, the published male fruit-fly central nervous system, is stepping inside a Flappy-style game. Spikes happen. Descending neurons move. Sometimes the fly clears the pipes.

It can do that because other people already learned which cells answer to vision, to touch and wind, and to smell, and because I picked a few visual ones and wrote a translator. The scan made the wires available. The stimulus maps made them mean something. The game, the teacher, and the readout did the rest.

That is a smaller claim than "I uploaded a fly and it plays." It is also the claim that survives contact with the code.

---

*Project: Flappy Fly. Connectome: MaleCNS v1.0, Berg et al., Cell (2026), via the `flybrain` simulator by alextitonis. The biological fly did not learn Flappy Bird. No Flappy Bird artwork is in the game.*
