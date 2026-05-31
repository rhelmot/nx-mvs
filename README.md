nx-mvs
======

An extension of `Maximum Convex Subgraphs Under I/O Constraint for Automatic Identification of Custom Instructions`. Adds Python bindings for working with NetworkX graphs as well as a handful of features.

- Normally, convex subgraphs must have at least one output. We extend this to allow zero-outputs.
- We add controls to enforce connectivity of subgraphs.
- We add a sampling mode to resolve cases where the number of output subgraphs is so large that just enumerating them does not terminate in a reasonable timespan. This attempts to find a representative subset of the output graphs, discarding "low quality" results which are similar to others.
- Correspondingly, we add a "growth" mode, which attempts to recover unsampled graphs in the neighborhood of a particular sampled one. The intuition here is that downstream applications might (and in my case, do) have a property where not all subgraphs are useful, but the ones that are have subgraph relationships among them. This means that if we spot a relevant sampled subgraph, we can grow it into other relevant subgraphs which were not sampled.

### Warning

This repo is slop. I needed this implementation as a dependency for something and do not ahve the time to do this right so I had gpt5.4 do it for me. I really wouldn't trust this code as a correct implementation of anything without seriously examining its output for your particular application. I am also deeply apologetic to the author of the original mvs repo for torching your code like this. If you see this, you may wish to try to redo the above listed features for yourself - get in touch with me if you wanna hear about the application.

### License

Accordingly, I put this code into the public domain. As the old prophets once said, go nuts, show nuts, whatever.
