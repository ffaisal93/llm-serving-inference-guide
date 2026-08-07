# Topic 12: The API Layer

Every topic before this one has been about the model server.

How to get tokens out of a GPU quickly, how to batch them, how to shard a model that does not fit, how to put the whole thing on Kubernetes and let it scale itself.
That work is real and it is hard and it is not your product.

Your product is an HTTP endpoint.

Between the person typing a question and the vLLM process that answers it sits a service that nobody drew on the architecture diagram, and it is the service that decides whether your system survives a Tuesday.
It authenticates the caller.
It checks whether that caller is allowed to spend what this request will cost.
It looks to see whether somebody already asked this exact question four minutes ago.
It decides, when the fleet is saturated, who waits and who gets told no.
And it writes down what happened, in enough detail that you can bill for it and debug it.

That service is the subject of this topic.

## Why it gets skipped

Because it is boring until it isn't.

The first version of every LLM product is a thin proxy: take the request, forward it to the model, stream the answer back.
It works beautifully for weeks.
Then one of four things happens.

A customer writes a retry loop with no backoff, and their bug becomes everyone's outage.
Someone posts your demo somewhere popular, and the GPU fleet — which autoscales, which you tested, which is fine — cannot scale in the ninety seconds you have before every request times out.
Finance asks what a customer costs to serve and you cannot answer, because you never metered anything at the granularity of a customer.
Or you notice that the top forty questions account for a third of your traffic and you have been paying full price to answer each of them from scratch, thousands of times a day.

None of those are model problems.
All four are solved in the same place, by the same service, and it is the one you did not build.

## What this topic covers

**Rate limiting**, done properly.
The five algorithms — fixed window, sliding window log, sliding window counter, token bucket, and the leaky-bucket/GCRA family — with what each one actually costs to store and when you would pick it.
Then the part generic articles get wrong: for an LLM API the *request* is the wrong unit of measurement, because one request can be two hundred tokens or two hundred thousand.
You will meter tokens, concurrency, and dollars, not requests.

**Distributed enforcement.**
Why an in-process counter is silently wrong the moment you run two replicas, how to do it in Redis without the race condition that everyone ships first, and when approximate local limiting is the right call anyway.

**Queuing and admission control.**
What to do at capacity when rejecting is the wrong answer — bounded queues, backpressure, priority classes, load shedding — and why an unbounded queue is not a safety valve but a slower way to fail.

**Caching**, in layers.
Exact-match, semantic (nearest-neighbour lookup on an embedded query, with the false-positive risk explained honestly rather than glossed over), embedding caches, and provider-side prompt caching — which is a genuinely different mechanism from your own cache, and composes with it rather than replacing it.
Plus the operational half: key design, TTLs, invalidation, cache stampede, and what a hit rate is telling you.

**Gateway patterns.**
Whether this logic belongs in your application, a sidecar, or a dedicated gateway like Envoy or Kong — with the tradeoffs stated plainly — along with key management, logging, and multi-provider failover.

## What you will be able to do

Explain, at a whiteboard, why a fixed-window limiter lets through twice its stated limit and exactly when that matters.

Write a token-metered limiter that charges an estimate up front and reconciles against the real usage when generation finishes, because you cannot know a request's cost until after you have served it.

Build a two-tier cache that tries exact match, falls back to semantic similarity, refuses a near-miss that would have returned a confidently wrong answer, and reports its hit rate.

Choose a similarity threshold on evidence rather than vibes — and say out loud what your false-positive budget is.

Answer two of the four questions that come up in nearly every AI-engineering system design interview: *how would you handle rate limits for millions of users?* and *how would you cache LLM responses efficiently?*

## How this fits with the rest of the guide

This topic sits above the serving layer and below the application.

When capacity is the real constraint, the answer is in this guide's own autoscaling chapters, and the signals you need to see any of this are in its monitoring chapters — this topic points there rather than reprinting them.
For the layer *above* — an agent loop that decides on its own how many model calls to make, and how you keep that from costing forty dollars a run — see the `learn-production-agent` book's treatment of budgets and cost control.
That book charges a budget before spending it; this topic is where the same idea is enforced against a stranger on the internet who does not share your interests.

Read the deep dive with a terminal open.
The worked example is dependency-free Python and the output in the chapter is real output from running it.
