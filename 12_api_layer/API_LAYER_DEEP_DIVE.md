# The API layer: rate limiting, quotas, and caching

There is a service in your architecture that nobody drew.

You have a diagram with a load balancer, a Kubernetes cluster, some GPU nodes running vLLM, a monitoring stack off to the side.
Between the arrow labelled "user" and the box labelled "inference" there is a gap, and in production something has to live in that gap.

That something is your API layer, and it is the actual front door of your product.

The model server is not the front door.
vLLM does not know who your customers are, has no opinion about whether this particular caller has spent their monthly allowance, will not notice that the same question arrived eleven times in the last minute, and — critically — will happily accept every request you hand it right up to the point where its queue depth makes the ninety-ninth percentile latency worse than a timeout.
It is a very good engine.
It is not a car.

The API layer authenticates, authorizes, meters, caches, queues, and degrades.
Six verbs.
This chapter is about all six, with the most time spent on the two that come up in every design review and every interview: metering, which people call rate limiting, and caching.

## What breaks when this layer is missing

Three failures, and you will see all three if you run long enough.

**One user starves everyone.**
A customer ships an integration with a retry loop, no backoff, and a bug — a malformed request that gets a 400, retried immediately, forever.
On a normal REST API that is a nuisance in the logs.
On an LLM API it is an outage, because their requests occupy GPU slots that are genuinely scarce, and your other customers experience it as everything getting slow at once.
Without per-caller metering, one bad script is indistinguishable from organic growth until you read the access logs by hand.

**A viral moment melts the fleet.**
Autoscaling works, and for GPU workloads it is slow: a node, a container image measured in gigabytes, then model weights measured in more gigabytes.
Two to ten minutes of cold start is normal, and this guide's autoscaling chapters cover shaving that down.
None of it helps in the ninety seconds between someone popular linking to your demo and your request pool being exhausted.
Something has to absorb that spike or shed it deliberately.

**You answer the same question at full price, forever.**
Every consumer-facing LLM product has a fat head in its query distribution — "what can you do", "how do I cancel", the same onboarding prompt from every new user, the same document summarized by six people on the same team.
Each is a fresh forward pass through a very large model, paying for a computation whose answer you already have from four minutes ago.

All three have the same shape: a decision that must be made *before* the request reaches the model, using state the model server does not have.

---

## Rate limiting

Rate limiting is the practice of capping how much of a shared resource any one caller can consume in a window of time.

The word *limiting* undersells it.
What you are really doing is deciding, continuously and automatically, how a finite resource gets divided among callers who each want all of it.
That is an allocation policy, and the algorithm you pick determines the shape of the traffic your backend sees.

There are five algorithms worth knowing.
They differ in exactly two ways that matter: how much state they cost you per caller, and what burst behaviour they permit.

### Fixed window counter

Keep one integer per caller per time window.
Round the current time down to the window — say, the current minute — and use that as part of the key.
Increment on each request; if the counter exceeds the limit, reject.
When the clock ticks over to the next minute, the key changes and the count starts at zero.

This is the simplest thing that works, and its cost is one small integer per active caller, which is essentially free.

It has one flaw, and it is a real one: **the boundary burst**.
A caller limited to 100 requests per minute can send 100 at 11:59:59 and another 100 at 12:00:00 — two hundred requests in a two-second span, both windows individually legal, and your backend seeing double the rate you configured.

Whether that matters depends on what you are protecting.
For a limit that exists mostly to catch runaway scripts, a 2× overshoot for one second is nothing.
For a GPU fleet sized with 20% headroom, it is an incident.
Assume LLM serving is the second case unless you have measured otherwise.

### Sliding window log

Store a timestamp for every request the caller makes.
On each new request, drop the timestamps older than the window, count what remains, and admit if the count is under the limit.

This is exactly correct — no boundary artefact, because there is no boundary.
The window slides continuously with the current time.

The cost is one entry per request per caller for the length of the window.
A caller allowed 10,000 requests per minute needs up to 10,000 timestamps held for a minute, and you pay that for every active caller simultaneously.
In Redis this is a sorted set, pruned with `ZREMRANGEBYSCORE` and counted with `ZCARD`.

Use it when limits are small and precision matters — expensive operations, per-account write limits, anything where "approximately 100" is not acceptable.
Do not use it as the front-line limiter for high-volume traffic.

### Sliding window counter

The compromise, and in practice the most common choice.

Keep two fixed-window counters, current and previous, and estimate the sliding rate as a weighted blend:

\\( \\text{estimate} = c_{\\text{cur}} + c_{\\text{prev}} \\times (1 - f) \\)

where \\( f \\) is the fraction of the current window elapsed.
A quarter of the way into the current minute you count this minute's requests in full plus 75% of last minute's.

The estimate assumes the previous window's requests were spread evenly through it, which is not true, so it is approximate.
Cloudflare measured this across 400 million requests from 270,000 sources and found 0.003% of requests wrongly allowed or limited — approximate, but not by much.
Two integers per caller, no boundary burst, and for most general-purpose API rate limiting this is the right default.

### Token bucket

Now we get to the one you will actually implement.

Picture a bucket that holds up to \\( B \\) tokens and refills at \\( r \\) tokens per second.
Each request removes tokens equal to its cost.
If the bucket does not have enough, the request is rejected.

Two parameters, two meanings, and keeping them straight is most of the skill:

- \\( r \\) is your **sustained rate**. Over the long run, no caller gets more than \\( r \\) per second.
- \\( B \\) is your **burst allowance**. A caller who has been idle can spend the full bucket at once.

That second property is the whole reason to choose token bucket.
Real clients are bursty and their burstiness is usually legitimate — a user opens the app and five parallel requests fire to populate the screen.
A limiter that smooths those into a queue makes the product feel slow to protect a backend that could have handled them.
Token bucket says: go ahead, you have been quiet, spend your savings.

You do not need a background process to refill, which is the mistake in most naive implementations.
Store the token count and the timestamp of the last update; on each request add \\( \\Delta t \\times r \\) tokens, capped at \\( B \\).
Lazy refill, exact, two numbers per caller.

Token bucket is the default for LLM APIs, and the reason is in the next section: it is the only one of these five that naturally handles requests with *different costs*.

### Leaky bucket and GCRA

Leaky bucket inverts the picture.
Requests pour into a bucket, the bucket drains at a fixed rate, and if it overflows requests are rejected.
The output rate is constant regardless of the input, which makes it a *shaper* rather than merely a limiter — useful when your backend needs smooth input.
NGINX's `limit_req` is a leaky bucket; its `burst` parameter sets the queue depth and `nodelay` forwards queued requests immediately rather than spacing them.

The elegant version is **GCRA**, the Generic Cell Rate Algorithm, borrowed from ATM networking.
Instead of a token count it tracks one timestamp: the **theoretical arrival time** (TAT), the earliest moment at which the next request would be perfectly conforming.

Two constants — the **emission interval** \\( T \\), the ideal gap between requests, which is the window divided by the quota; and the **delay variation tolerance** \\( \\tau \\), which is burst capacity expressed as time.
A request at time \\( t \\) is admitted when

\\( t \\ge \\text{TAT} - (\\tau + T) \\)

and on admission you set \\( \\text{TAT} \\leftarrow \\max(t, \\text{TAT}) + T \\).

One timestamp per caller, no drip process, no counter to reset.
As Brandur Leach puts it in the canonical write-up, removing the drip removes a whole category of failure, because an offline or overloaded drip worker limits *incorrectly* rather than merely limiting late.

GCRA and token bucket are close cousins — a token bucket with a fractional count carries the same information as a TAT.
Pick GCRA for minimum state and exact pacing; pick token bucket when costs vary per request, which for LLM traffic they emphatically do.

### The comparison, condensed

| Algorithm | State per caller | Boundary burst | Variable cost | Good for |
|---|---|---|---|---|
| Fixed window | 1 integer | Yes, up to 2× | Awkward | Cheap coarse protection |
| Sliding window log | 1 entry per request | No | Yes | Small, expensive limits |
| Sliding window counter | 2 integers | No | Awkward | General-purpose default |
| Token bucket | 2 numbers | Controlled by \\( B \\) | Natural | **LLM APIs** |
| GCRA / leaky bucket | 1 timestamp | Controlled by \\( \\tau \\) | Possible, less natural | Smooth pacing, minimum state |

---

## The LLM twist: requests are the wrong unit

Here is where the generic rate-limiting article stops being useful.

Every one of those algorithms, as usually presented, counts requests.
For an LLM API, counting requests is close to meaningless, because the variance in what a request costs is enormous.

One request is "hi" and produces twelve tokens.
Another is a 180,000-token document with "summarize this" on the end, which costs roughly four orders of magnitude more compute, occupies a KV-cache allocation the size of a small dataset, and holds a slot on a GPU for thirty seconds.
Both are one request.

Limit on requests and you have built something that either rejects the trivial user unnecessarily or lets the expensive one consume the entire fleet.
So: **meter what is scarce**.

**Tokens.**
This is the primary axis. Limits look like "60,000 input tokens per minute, 20,000 output tokens per minute," which is exactly how the commercial providers express theirs — OpenAI, Anthropic and the rest publish TPM (tokens per minute) alongside RPM (requests per minute), and it is TPM you hit first.

**Concurrent requests.**
The second axis, and the one people forget. Tokens per minute does not bound how many of a caller's requests are simultaneously occupying GPU slots. A caller running fifty parallel long generations may be within their token budget over a minute while pinning your batch scheduler right now. Cap in-flight requests per caller separately.

**Dollars.**
The third axis, for anything with per-seat pricing or a free tier. Tokens are a proxy for cost, but the exchange rate differs by model: a thousand tokens on your largest model may cost fifteen times a thousand tokens on the small one. If you route between models, meter money, or callers will discover that routing to the expensive model is free.

### Charging before you know the price

There is a real difficulty here and it is worth naming clearly.

You cannot know a request's token cost until it has finished.
Input tokens you can count before dispatch — that is just tokenization.
Output tokens are unknown until generation stops.

The standard resolution is a two-phase charge:

1. **Reserve** on admission, using input tokens plus an estimate of output. The estimate can be the caller's `max_tokens` (conservative, and it will make heavy users feel over-limited), a rolling percentile of that caller's historical output length (better), or a per-endpoint constant (fine to start).
2. **Reconcile** on completion. Charge the difference if you underestimated; refund it if you overestimated.

Reserving at `max_tokens` and never reconciling is the naive version, and it is why some providers feel far stingier than their published numbers suggest.
Reconciliation is maybe fifteen lines of code and it materially changes how the product feels.

The worked example below implements exactly this.

### Whose limit is it

A single limit keyed on one identity is not enough, because you have several failure modes to prevent and they have different scopes.

**Per API key** catches the runaway script. One customer, three integrations, three keys; when one integration misbehaves you want the other two unaffected. This is your isolation boundary.

**Per user** matters in a consumer product where identity is a person, not a key, and it is what stops one enthusiastic user from consuming a shared team allowance.

**Per organization or account** is where the commercial limit lives — what the customer bought. It caps the sum across all their keys and users.

**Global** is the one people leave out and then wish they hadn't. A ceiling on total admitted load, expressed in whatever units your fleet is actually sized in, that exists to protect the fleet from the aggregate of everyone behaving legitimately at once.

Evaluate them cheapest-first and reject on the first failure, so a request that is going to be denied for concurrency does not first burn a Redis round trip against the token budget.

**Tiers** map onto this straightforwardly. Free, pro, enterprise get different \\( r \\) and different \\( B \\), and it is worth understanding that they are separate knobs: raising \\( B \\) alone gives a tier a better *feel* — snappier bursts, no perceived throttling on normal use — without giving away any additional sustained throughput. That is often the cheapest upgrade you can ship.

Give new accounts a lower ceiling that grows with account age and payment history.
Abuse is disproportionately from new accounts, and this costs nothing to implement.

### What to return

When you reject, be useful about it.

Status **429 Too Many Requests**.
Not 503, which says your server is broken; not 403, which says the caller may never do this.
429 says: this is a valid request, you are over quota, try again.

Include **`Retry-After`**, in seconds, and compute it honestly from the limiter state.
You know exactly when the bucket will hold enough tokens — it is \\( (\\text{cost} - \\text{tokens}) / r \\).
Returning a real number instead of a hardcoded 60 is the single highest-leverage thing you can do for client behaviour, because well-written clients will honour it and stop hammering you.

For the limit state itself there is now a standards-track answer.
The IETF HTTPAPI working group's draft `RateLimit header fields for HTTP` (draft-11 as of May 2026) defines two Structured Field response headers:

```
RateLimit-Policy: "burst";q=100;w=60,"daily";q=1000;w=86400
RateLimit: "default";r=50;t=30
```

`q` is the quota, `w` the window in seconds, `r` the remaining quota, `t` the seconds until reset.
Crucially for us, there is a `qu` parameter for the **quota unit** — so `qu="tokens"` says plainly that this is a token budget, not a request budget.
That is the first standard that expresses what an LLM API actually needs.

The older convention — `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` — is what most clients still parse.
Emit both for now.
The draft is still a draft; it is also clearly where this is going.

Return the same headers on **successful** responses too.
A client that can see it has 400 tokens left out of 60,000 can slow down before it gets rejected, which is better for both of you than discovering the limit by hitting it.

Two more things worth doing:

**Never reject a streaming response mid-stream because of reconciliation.** If the generation overshot its estimate, charge it and let the *next* request pay the price. Truncating an answer the user is watching arrive is a terrible experience in exchange for a rounding error.

**Distinguish "you are over quota" from "we are over capacity."** They look the same to the client and require different responses — the first is the caller's problem and backing off fixes it, the second is yours and the caller backing off merely spreads the pain. Different error codes in the body, even if both are 429.

---

## Distributed rate limiting

Everything above assumed one process holding the state.

The moment you run two replicas, an in-process limiter is wrong, and it is wrong in a specific and predictable direction: with \\( N \\) replicas behind a round-robin balancer, each enforcing the full limit locally, a caller can consume up to \\( N \\times \\) the limit.

You cannot fix this by dividing the limit by \\( N \\).
Traffic is not evenly distributed across replicas — connection reuse and keep-alive see to that — so a caller whose connections happen to land on one replica gets \\( 1/N \\) of what you promised them, and you have turned an over-limit bug into an under-limit bug that generates support tickets.

The state has to be shared.

### Doing it in Redis, correctly

Redis is the standard answer: fast, single-threaded, and with a data model that fits.

The trap is that every one of these algorithms is a **read-modify-write**.
Read the current tokens, decide, write the new tokens.
Issued as separate commands, two concurrent requests can both read a bucket holding one token, both conclude they are allowed, and both write back zero.

Two requests admitted, one token spent.
Under load this is not rare; it is the common case, and it scales with your concurrency, which means the limiter fails hardest exactly when it matters most.

`INCR` is atomic and `EXPIRE` is atomic, but `INCR` *then* `EXPIRE` is not, and the failure there is subtle and worse: if the process dies between them you have a counter with no TTL, which never resets, and that caller is permanently limited.

The fix is to make the whole read-decide-write sequence one atomic unit.
In Redis that means a Lua script via `EVAL` (or `EVALSHA` after loading it once), which runs to completion without interleaving.
Redis's own rate-limiting tutorial is explicit that every one of its five limiter recipes uses Lua for precisely this reason.

Three practical notes.

**Keep the script small and pure.** It blocks the entire server while it runs. Do arithmetic, not iteration over large collections.

**Pass the timestamp in as an argument, not from `TIME` inside the script.** Scripts that call `TIME` are non-deterministic, which historically complicated replication; passing the caller's clock also lets you write tests with a fake clock, which you want.

**Declare your keys in `KEYS`, not `ARGV`.** Redis Cluster routes by key, and a script that touches an undeclared key will break the day you shard. When one logical limit spans multiple keys, use a hash tag — `{user:42}:tokens` and `{user:42}:reqs` — to force them onto the same slot.

### The accuracy-versus-latency tradeoff

A Redis round trip is somewhere between 0.2 and 2 milliseconds.

For an LLM request that will take three seconds, that is free — genuinely, do not think about it.
For the local rate limiter in front of a gateway handling a hundred thousand requests per second of mixed traffic, it is not free, and it is a hard dependency on a service that can be down.

The standard answer is two tiers, and it is what Envoy does by design.

A **local** limiter in each process — an in-memory token bucket, no network — set generously, whose job is to absorb obvious floods and shed them at zero cost.
Behind it, a **global** limiter in Redis that enforces the real quota.
Envoy's documentation makes this explicit: local rate limiting is used alongside global rate limiting to reduce load on the global service.

You accept some slop. A caller can exceed the global limit by roughly the sum of local burst allowances before the global limiter catches them, which is bounded and small if you configure it so.

The other technique worth knowing is **distributed token leasing**: each replica leases a block of quota from Redis — say 500 tokens — and spends it locally, returning for another block when it runs low. One round trip per 500 tokens instead of one per request. The cost is that unreturned leases held by a crashed replica are lost until they expire, so the effective limit is slightly under the configured one.

And decide, explicitly and in advance, what happens when Redis is unreachable.

**Fail open** admits everything, and your rate limiter's outage does not become your API's outage. This is usually right for a limiter protecting against accidents.

**Fail closed** rejects everything, which turns a Redis blip into a total outage but prevents an attacker from disabling your limits by taking down one dependency. Right for a limiter that is a security control.

Most LLM APIs should fail open on the abuse limiter and fail closed on the billing limiter.
Whatever you choose, choose it deliberately and put it in a config flag, because you will want to flip it during an incident.

---

## Queuing and admission control

Rate limiting answers "is this caller allowed to spend this?"
Admission control answers a different question: "do we have the capacity to serve this right now?"

They are independent.
A caller can be well within quota while the fleet is saturated, and rejecting them with a 429 tells them to slow down when the honest answer is "everyone should slow down, briefly."

### The queue

The instinct is to queue: hold the request until a slot frees up.

For LLM serving this is more attractive than for most workloads, because your requests are already long — a user who waits three seconds will not notice four.
Queueing converts a hard rejection into slightly worse latency, which is a very good trade up to a point that arrives faster than you expect.

**An unbounded queue is not a safety valve.**
It is a mechanism for converting a fast failure into a slow one.

Requests arrive at a rate you do not control and are served at a rate fixed by your GPU count.
When arrival exceeds service the queue grows without limit, and every request in it still gets served — eventually, by which time the client has timed out and retried.
So you burn GPU on answers nobody will read, for clients that have already asked again.
Memory grows, percentiles go vertical, and the system does not recover on its own after the surge passes, because it is working through a backlog while new traffic keeps arriving.

Three rules.

**Bound the queue.** Set a maximum depth and reject with 429 or 503 when it is full. Size it from Little's Law: with a target wait \\( W \\) and a service rate \\( \\lambda \\), the depth is \\( L = \\lambda W \\). If you serve 20 requests per second and will tolerate 2 seconds of queueing, your queue holds 40. Not 10,000.

**Bound the wait.** Stamp each request on arrival and drop it when it has been waiting longer than the client's timeout. Serving a request whose caller has given up is pure waste, and under load it is the waste that keeps you from recovering.

**Watch depth, not just latency.** Queue depth is a leading indicator; latency is a lagging one. By the time p99 moves, the queue has been growing for a while. This is the signal to wire into the autoscaler, and the guide's monitoring chapters cover getting it out of the serving layer.

### Priority classes

Not all requests deserve the same treatment, and under pressure you should say so.

A reasonable default set: **interactive** (a human is watching a cursor blink), **background** (a batch job, a nightly summarization), and **best-effort** (speculative prefetch, cache warming, evaluation runs).

Under load, interactive traffic goes to the front.
Background work waits, and that is correct — nobody is watching.
Best-effort is dropped entirely, and if you have built it properly, nothing breaks.

The failure mode to design against is starvation: a class that never gets served because a higher class is always full.
Weighted fair queueing rather than strict priority — background gets 20% of slots even when interactive is saturated — avoids it, and the cost is a slightly worse tail for interactive traffic in exactly the conditions where the tail is already bad.

### Load shedding

Shedding is deciding, deliberately, to drop work you could technically accept, because accepting it makes everything else worse.

Shed by cost first.
A 100,000-token summarization occupies a GPU for thirty seconds; ten short chats fit in the same window.
When you are saturated, rejecting the one expensive request preserves service for ten users instead of one.
This feels unfair and it is the right call, and you should document the policy so support can explain it.

Shed by tier second — free before paid, best-effort before interactive.

And shed *early*, at the edge, before you have spent anything on the request.
A request rejected after tokenization, embedding, and a cache lookup has already cost you real work.

### Degradation

The most under-used tool here.

Instead of rejecting, serve something cheaper.

Route to a smaller model — the guide's chapters on multi-model serving cover keeping a Haiku-class model warm alongside your large one, and under load, routing overflow traffic to it is a far better experience than a 429.
Drop optional stages: skip the reranker, skip the second retrieval pass, cut `max_tokens`.
Serve a stale cache entry rather than nothing.
Fall back to a non-generative path where one exists — a search results page instead of a synthesized answer.

Degradation has to be designed in.
You cannot add it during an incident.
Decide the ladder now, put each rung behind a flag, and test that the flags work.

---

## Caching

Caching is where the API layer stops being a tax and starts being the reason your unit economics work.

There are four distinct things people call "the cache" and they operate at different layers with different semantics.
Conflating them is the source of most confused conversations about this topic, so let us separate them first.

### The four layers

**1. Exact-match response cache.** You hash the request and store the response. Same request, same answer, no model call. Yours, in your infrastructure.

**2. Semantic cache.** You embed the query and look for a *similar* previous query. Different text, similar meaning, reuse the answer. Yours.

**3. Embedding cache.** You cache the embedding vectors themselves, because computing them is a model call too. Yours, and the least glamorous of the four while frequently having the best return.

**4. Provider-side prompt cache.** The model provider caches the *internal computation* for a repeated prompt prefix. Theirs, not yours, and it is not a response cache at all.

### Exact-match

The boring one, and the one to build first.
Normalize the request, hash it, look it up; on a miss, call the model and store the result.

It handles more traffic than you expect, because a great deal of LLM traffic is genuinely identical — the same system prompt over the same document, the same button in your UI firing the same templated request, the same FAQ question typed the same way by different people.

And it is exact, so it cannot be wrong.
If the key matches, the request matched.
That property is worth more than it sounds, and everything below trades some of it away.

The subtlety is **normalization**: what counts as "the same request."
Lowercase and strip whitespace, certainly.
But `temperature` must be in the key — a request at temperature 1.2 is asking for variety, and serving it a cached answer defeats the point, which is a good argument for not caching non-zero-temperature requests at all unless you have thought about it.
So must the model name, `max_tokens`, the system prompt, the tool definitions, and anything else that changes the output distribution.

### Semantic caching

Here is where it gets interesting, and where it gets dangerous.

"How do I reset my password?" and "I forgot my password, what do I do?" are the same question.
An exact-match cache sees two unrelated strings.
A semantic cache embeds both into vectors, notices they are close, and serves the cached answer for the second.

The mechanism: embed the incoming query, do a nearest-neighbour search against the embeddings of cached queries, and if the closest one exceeds a similarity threshold, return its cached response.
GPTCache, the reference open-source implementation, is built on exactly this pipeline — an embedding function, a vector store, a similarity evaluator, and an eviction policy.

The appeal is obvious. Hit rates that exact matching cannot touch, on the head of your query distribution where it matters most.

**Now the honest part.**

A semantic cache can return the wrong answer — not a stale answer, a *wrong* one, to a question that was never asked.

The threshold is doing all the work, and the assumption underneath it is that embedding similarity tracks answer equivalence.
It approximately does, not reliably, and the failures are not random — they cluster on exactly the short, near-identical, high-stakes queries a cache sees most.
Consider:

- "What is the refund window for **annual** plans?" versus "for **monthly** plans." Nearly identical vectors. Completely different answers. Both are policy statements a customer will act on.
- "Should I take this medication with food?" versus "Should I take this medication **without** food?" Negation is close to invisible to a bag-of-meaning embedding.
- "What's my account balance?" asked by two different users. Identical vectors, and if your key does not scope by user, you have just built a data leak with excellent latency.

That last one is not a quality problem, it is a security incident, and it is covered properly in the key design section below.

Research has caught up with this.
The **vCache** work (arXiv 2502.03771) analyses static thresholds directly and finds that the similarity distributions of correct and incorrect cache hits *overlap substantially* — meaning no single global threshold cleanly separates them, and the "right" threshold varies per query and per embedding model.
Their proposed alternative learns a per-embedding threshold online against a user-specified error bound, reporting substantially higher hit rates at much lower error rates than static thresholds.
There is also adversarial work on deliberately crafting queries that collide with cached entries under a given threshold.

The practical guidance:

**Set the threshold from labelled data, not intuition.** Take a few hundred real query pairs, label whether the answers should be the same, compute similarities, and look at where the distributions overlap. That plot is the most useful thing you can produce about your cache.

**Scope the cache narrowly.** Semantic caching is safe on a documentation FAQ bot and reckless on anything giving personalized, financial, medical, or legal answers. It is a per-surface decision, not a global switch.

**Never cache semantically across tenants.** Partition the vector index by tenant. Cheaper to enforce than to explain afterwards.

**Log every semantic hit with its similarity score.** When you get a complaint about a wrong answer, that log is how you find out the cache did it. Without it you will spend a week blaming the model.

**Consider a cheap verification step.** A small fast model asked "do these two questions have the same answer?" costs a fraction of the large generation you are avoiding, and converts an unbounded risk into a bounded cost. This is a real pattern and it is under-used.

**Have a false-positive budget and measure against it.** If you cannot say what rate of wrong answers you will accept, you are not ready to run a semantic cache.

### Embedding cache

Embeddings are model calls.
For a semantic cache, every incoming query needs one — you cannot look up without embedding first.
So the thing that makes your cache fast is itself a per-request model call, and if you do not cache it, your semantic cache has a fixed floor on both latency and cost.

Cache embeddings by content hash.
They are deterministic for a given model and input, so this is exact-match caching with none of the semantic risk, and the hit rate on a repetitive workload is very high.

Include the embedding model name and version in the key.
Vectors from different models are not comparable, and mixing them silently produces a similarity metric that means nothing.
When you upgrade the embedding model you must rebuild the whole index — treat it as a migration, not a config change.

### Provider-side prompt caching

This is a different mechanism, and the confusion between it and your response cache is worth clearing up carefully.

Your response cache stores *outputs*.
Provider prompt caching stores *intermediate state* — the key-value attention tensors computed while processing a prompt prefix.
It does not skip generation; it skips re-processing the part of the input the model has already seen.

Four consequences follow.

It is **prefix-based**. The cache matches on an exact prefix, so static content must come first and variable content last. Anthropic's hierarchy is tools, then system, then messages, and placing a breakpoint after something that changes per request means it never hits.

It has **minimum sizes**. Anthropic requires 512 tokens for Opus 5-class models and 1,024 or more for others; OpenAI requires 1,024. Below that nothing is cached and no error is returned — a silent no-op worth checking for in your usage metrics.

It has **explicit pricing**. On Anthropic a 5-minute cache write costs 1.25× base input, a 1-hour write 2×, and a read 0.1× — so it pays for itself on the second hit. OpenAI's newer models similarly charge 1.25× for writes with a large read discount and a minimum 30-minute retention.

It is **controlled differently per provider**. Anthropic is opt-in via `cache_control` markers (up to four explicit breakpoints, or automatic placement); OpenAI's is automatic for eligible prompts with an optional `prompt_cache_key` to improve routing; Gemini offers both implicit and explicit context caching.

And it **composes with your cache rather than competing**:

1. Your exact cache hits — no provider call at all. Cheapest.
2. Your semantic cache hits — no provider call. Cheap, with the risk described above.
3. Miss, so you call the provider — and the provider's prompt cache makes *that call* cheaper by reusing your long system prompt and retrieved documents.

Layer 3 helps every request that reaches it, including all your cache misses, and it requires no infrastructure from you.
Structure your prompts prefix-stable and it costs you a code review.

One more thing worth knowing: **KV cache** inside your own vLLM deployment is the same idea, applied locally. vLLM's automatic prefix caching reuses attention state across requests that share a prefix. If you self-host, that is your version of provider prompt caching, and this guide's vLLM chapters cover how to enable and size it.

### Cache key design

The most consequential fifteen lines in the whole system.

The key must include everything that changes the answer:

- the **normalized prompt** (lowercased, whitespace-collapsed, with your specific normalizations)
- the **model identifier** including version — a new model version is a new cache namespace
- the **sampling parameters** — temperature, top_p, max_tokens, stop sequences
- the **system prompt** or its hash
- the **tool definitions** if any
- the **prompt/template version**, so a prompt change invalidates the cache automatically rather than serving answers generated under the old instructions

And then the one that matters most:

**Include the tenant, and include the user whenever the response is personalized.**

If your responses depend on who is asking — retrieved documents scoped by permission, account data, anything from a per-user context — then a key without the user identity means user B can receive user A's answer.

This is not a subtle bug.
It is a cross-user data leak, it will be found, and it will be found by a customer.
It is also, in my experience, the single most common serious defect in hand-rolled LLM caches, because it does not show up in testing (one test user), it does not show up at low traffic (few collisions), and it produces no error when it fires.

The rule that makes it hard to get wrong: **derive the cache key from the same context object that produced the prompt.**
If the retrieval was scoped by `user_id`, then `user_id` is part of the input, and it belongs in the key mechanically rather than by remembering.

Where responses genuinely are not personalized — a public documentation bot — sharing across users is exactly the point, and you get much better hit rates.
Make that a deliberate, per-surface, written-down decision.

### TTL policies

TTL is a bet about how long an answer stays correct.

Set it from the volatility of the underlying data, not from a habit:

- **Static knowledge** — explanations, definitions, code examples: hours to days.
- **Documentation-grounded answers**: tie the TTL to your documentation deploy cycle, or better, invalidate on deploy.
- **Data-grounded answers** — anything reading a live database: minutes at most, and consider whether caching is appropriate at all.
- **Personalized answers**: short, and scoped per user.
- **Anything with a timestamp in the answer**: do not cache, or strip the timestamp.

Two refinements.

**Jitter the TTLs.** Entries written together expire together, and a synchronized expiry is a self-inflicted stampede. Add ±10% randomness.

**Consider adaptive TTLs.** Extend on hit — a popular entry stays warm — and expire cold entries fast. This approximates LFU eviction with much simpler bookkeeping.

### Invalidation

TTL is invalidation by giving up.
It is fine, and it is what you should do by default, but sometimes you need to actually invalidate.

**Version prefixes** are the pattern that works. Put a version in the key namespace — `cache:v7:...` — and to invalidate everything, bump to v8. Old entries are orphaned and expire on their own TTL. No scanning, no deletes, instant, and trivially reversible if you were wrong.

**Tag-based invalidation** for finer control. Record which source documents contributed to each cached answer; when a document changes, invalidate the entries tagged with it. This requires you to track provenance, which you probably want for citations anyway.

**Event-driven invalidation** for data-grounded answers: your CMS publishes a change event, the cache layer drops the affected keys.

**Do not scan for keys to delete.** `KEYS` blocks Redis, `SCAN` is a slow-motion version of the same problem. Design the namespace so that invalidation is a namespace change, not a search.

### Cache stampede

Here is the failure that turns a cache from a protection into an amplifier.

A popular entry expires.
A hundred concurrent requests for it all miss simultaneously.
All hundred call the model.
All hundred compute the same answer.
All hundred write it back.

You have just sent 100× your steady-state load at the GPU fleet, at the exact moment the cache stopped helping.
This is a **cache stampede**, also called a thundering herd or a dogpile, and for expensive backends it is the most important cache failure mode to design against.

Four mitigations, roughly in order of how often you should reach for them.

**Single-flight (request coalescing).** The first request to miss takes a lock and computes; the others wait on the same in-flight computation and share its result. One model call instead of a hundred. This is the primary defence and it is the one to implement first. Go's `singleflight` package is the canonical implementation of the idea; the pattern is a map from key to a pending future.

**Probabilistic early expiration (XFetch).** Rather than expiring at a hard boundary, each request nearing the TTL has a small and growing probability of recomputing early, so one lucky request refreshes the entry while everyone else is still being served the cached value. The classic formulation recomputes when

\\( t_{\\text{now}} - \\delta \\beta \\ln(\\text{rand}()) \\ge t_{\\text{expiry}} \\)

where \\( \\delta \\) is how long the recomputation takes and \\( \\beta \\) tunes eagerness.
It is elegant, it is a handful of lines, and it eliminates the synchronized-expiry class of stampede entirely.

**Stale-while-revalidate.** Keep serving the expired value while one background task refreshes it. Everyone gets a fast response; one of them is a few seconds stale. Whether that is acceptable is a product decision, and for most LLM answers it very much is.

**Pre-warming.** For entries you know will be hot — a launch, a scheduled campaign, the FAQ — populate the cache before traffic arrives.

Locks need care. A lock holder that dies must not block everyone forever, so set a TTL on the lock; a waiter must not wait longer than the client's timeout. Bound both.

### Measuring it

A cache you do not measure is a cache you do not have.

**Hit rate**, split by tier. Exact and semantic hit rates are different numbers with different meanings, and a single blended figure hides the one you need to watch.

What is good? It depends entirely on the workload, and anyone quoting a universal number is selling something. Some calibration:

- A **general-purpose assistant** with an open-ended query distribution: 5–15% is realistic. The tail is genuinely long. Do not be disappointed.
- A **documentation or support bot** with a fat head of common questions: 30–60% is achievable, mostly on the top few hundred queries.
- An **internal tool** with templated prompts over a bounded document set: 60%+ and it should be higher if it isn't.

The diagnostic signals matter more than the absolute number:

- **Hit rate near zero** means your key is over-specific. Something varying is in it that should not be — a timestamp, a request ID, a session token. Log a few keys and look.
- **Hit rate suspiciously high** on personalized content means your key is under-specific, and you should check for the cross-user leak *today*.
- **A sudden drop** means something changed the key shape: a model version bump, a prompt template edit, a new field in the context object. This is one of the most useful alerts you can configure.

Also track **cost avoided** — the token cost of the requests you served from cache, which is the number that justifies the system in a budget conversation — and **latency by path**, because cache hits should be single-digit milliseconds and if they are not, something is wrong with your store.

And track the semantic cache's **similarity distribution**.
Watching where hits cluster relative to the threshold tells you whether you have headroom or are living on the edge of it.

---

## Gateway patterns

Where does this logic live?

**In your application** — middleware in FastAPI or Express, same process as your business logic.
Simplest to start, full access to application context so a limit can depend on anything you know about the user, easy to test.
The cost is duplication: every service that needs limiting implements it and they drift, and your process spends CPU on requests it is about to reject.
Right when you have one service, or when policy genuinely needs application knowledge.

**In a sidecar** — a proxy container beside each application container, sharing a network namespace. Envoy in a service mesh is the canonical shape.
Policy is centrally managed and applied uniformly, in any language, without touching application code, and rejections happen before your app sees them.
The cost is operational: another container per pod, another config surface, another thing in the request path to debug at 3am.
Right when you have a mesh already.

**In a gateway** — one tier of proxies in front of everything: Envoy, Kong, APISIX, NGINX, or a cloud gateway.
One place for auth, limiting, routing and logging; rejection at the edge before any application resource is spent; and a battle-tested implementation you configure rather than write.
The cost is expressiveness.
Gateway limiting works on request attributes — headers, path, source IP, a JWT claim.
It does not know that this request will generate 8,000 tokens, and **that is precisely what an LLM API needs to limit on**.

So the honest recommendation is a split:

**Coarse limits at the gateway.** Requests per second per key, connection limits, IP-based abuse controls, body size caps. Cheap, fast, catches the obvious. Envoy's local rate limit filter is a token bucket per listener; its global filter calls out to a rate limit service over gRPC, with Envoy's own Go reference implementation backed by Redis. Kong and APISIX offer equivalents, with Kong's advanced plugin supporting sliding-window counters over a Redis strategy.

**Token-aware limits in your service.** Because only your service can tokenize the request, estimate the output, know which model it will route to, and reconcile after generation.

Do not contort a gateway into doing token accounting.
Do not skip the gateway because it cannot.

Three related concerns live at this layer too.

**Authentication and key management.** Store a hash of each API key, never the key — you should be unable to display it after creation. Prefix keys with an environment tag (`sk_live_`, `sk_test_`) so a leak is greppable and mistakes are visible. Support multiple active keys per account so rotation does not require downtime, and record last-used timestamps so you can find keys that are safe to revoke. Scope keys to permissions.

**Request and response logging.** You need it for debugging, billing, abuse investigation, and building evaluation sets from real traffic. You also have to reckon with the fact that prompts and completions frequently contain personal data. Log metadata always — token counts, latency, model, cache outcome, limiter decision. Log content selectively, with retention limits, redaction, and a per-customer opt-out. Sampling gives you most of the debugging value at a fraction of the exposure.

**Multi-provider routing and failover.** Once you have a layer between your app and the model, you can route: cheap models for easy requests, a different provider when your primary is degraded, a self-hosted vLLM for the bulk of traffic with a commercial API as overflow. Circuit breakers so a failing provider is removed quickly rather than eating your timeout budget on every request. This is also where you notice that the layer is doing quite a lot and should probably be its own service.

---

## A worked example

Everything above, in dependency-free Python, small enough to read in one sitting.

Two pieces.
A **token-metered limiter** that charges on tokens rather than requests, handles concurrency, and reconciles estimates against actuals — with the Redis Lua script it mirrors shown alongside, so you can see that the in-memory version is the same arithmetic without the network.
And a **two-tier cache** that tries exact match, falls back to semantic similarity, scopes by tenant, and reports its hit rate.

The mock embedder is a normalized bag of words.
It is not a real encoder and it is not pretending to be — its only job is to be deterministic, dependency-free, and to have the *shape* of a similarity function, so the threshold behaviour it demonstrates is real behaviour.
Swap in `sentence-transformers` and the surrounding code does not change.

```python
"""API layer demo: token-metered rate limiting + two-tier cache. Stdlib only."""

import hashlib
import math
import re
import time
from collections import OrderedDict

# ---------------------------------------------------------------- rate limiting

REFILL_LUA = """
local key        = KEYS[1]
local rate       = tonumber(ARGV[1])   -- tokens refilled per second
local capacity   = tonumber(ARGV[2])   -- bucket size (burst allowance)
local now        = tonumber(ARGV[3])   -- caller-supplied clock, seconds
local cost       = tonumber(ARGV[4])   -- tokens this request wants

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])
if tokens == nil then tokens = capacity; ts = now end

tokens = math.min(capacity, tokens + (now - ts) * rate)

local allowed = 0
if tokens >= cost then
  allowed = 1
  tokens  = tokens - cost
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) * 2)

local deficit = 0
if allowed == 0 then deficit = cost - tokens end
return {allowed, math.floor(tokens), deficit / rate}
"""


class TokenBucket:
    """In-memory mirror of REFILL_LUA. Same arithmetic, no network."""

    def __init__(self, rate_per_sec, capacity, clock=time.monotonic):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self.clock = clock
        self._state = {}

    def check(self, key, cost):
        now = self.clock()
        tokens, ts = self._state.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - ts) * self.rate)
        allowed = tokens >= cost
        if allowed:
            tokens -= cost
        self._state[key] = (tokens, now)
        retry_after = 0.0 if allowed else (cost - tokens) / self.rate
        return allowed, int(tokens), retry_after


class Limiter:
    """Three buckets per identity: tokens/min, requests/min, concurrency."""

    def __init__(self, tpm, rpm, max_concurrent, clock=time.monotonic):
        self.tokens = TokenBucket(tpm / 60.0, tpm, clock)
        self.requests = TokenBucket(rpm / 60.0, rpm, clock)
        self.max_concurrent = max_concurrent
        self.inflight = {}

    def admit(self, key, estimated_tokens):
        if self.inflight.get(key, 0) >= self.max_concurrent:
            return Decision(False, "concurrency", 1.0, 0)
        ok, left, retry = self.requests.check(key, 1)
        if not ok:
            return Decision(False, "requests", retry, left)
        ok, left, retry = self.tokens.check(key, estimated_tokens)
        if not ok:
            return Decision(False, "tokens", retry, left)
        self.inflight[key] = self.inflight.get(key, 0) + 1
        return Decision(True, None, 0.0, left)

    def settle(self, key, estimated_tokens, actual_tokens):
        """Reconcile the estimate against reality once generation finishes."""
        self.inflight[key] = max(0, self.inflight.get(key, 1) - 1)
        delta = actual_tokens - estimated_tokens
        if delta > 0:
            self.tokens.check(key, delta)          # overspend: charge it
        elif delta < 0:                            # underspend: refund
            t, ts = self.tokens._state[key]
            self.tokens._state[key] = (min(self.tokens.capacity, t - delta), ts)


class Decision:
    def __init__(self, allowed, reason, retry_after, remaining):
        self.allowed = allowed
        self.reason = reason
        self.retry_after = retry_after
        self.remaining = remaining

    def headers(self, policy_name, quota, window):
        h = {"RateLimit-Policy": f'"{policy_name}";q={quota};qu="tokens";w={window}',
             "RateLimit": f'"{policy_name}";r={max(0, self.remaining)};t={window}'}
        if not self.allowed:
            h["Retry-After"] = str(max(1, math.ceil(self.retry_after)))
        return h


# ---------------------------------------------------------------------- caching

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {"the", "a", "an", "is", "are", "do", "does", "i", "to", "of", "my", "in"}


def embed(text):
    """Deterministic bag-of-words unit vector. Stands in for a real encoder."""
    vec = {}
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        vec[w] = vec.get(w, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a, b):
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(v * large.get(k, 0.0) for k, v in small.items())


class TwoTierCache:
    def __init__(self, threshold=0.90, ttl=300.0, capacity=512, clock=time.monotonic):
        self.threshold = threshold
        self.ttl = ttl
        self.capacity = capacity
        self.clock = clock
        self.exact = OrderedDict()      # key -> (expires, response)
        self.semantic = []              # [(key, vector, expires, response)]
        self.stats = {"exact_hit": 0, "semantic_hit": 0, "miss": 0, "rejected": 0}

    def key(self, tenant, model, params, prompt):
        h = hashlib.sha256()
        for part in (tenant, model, repr(sorted(params.items())), prompt.strip().lower()):
            h.update(part.encode()); h.update(b"\x00")
        return h.hexdigest()[:16]

    def _expire(self):
        now = self.clock()
        for k in [k for k, (exp, _) in self.exact.items() if exp <= now]:
            del self.exact[k]
        self.semantic = [e for e in self.semantic if e[2] > now]

    def get(self, tenant, model, params, prompt):
        self._expire()
        k = self.key(tenant, model, params, prompt)
        if k in self.exact:
            self.exact.move_to_end(k)
            self.stats["exact_hit"] += 1
            return self.exact[k][1], "exact", 1.0
        scope = f"{tenant}|{model}"
        vec = embed(prompt)
        best, best_sim = None, 0.0
        for ckey, cvec, _exp, resp in self.semantic:
            if not ckey.startswith(scope):
                continue                                # never cross tenants
            sim = cosine(vec, cvec)
            if sim > best_sim:
                best, best_sim = resp, sim
        if best is not None and best_sim >= self.threshold:
            self.stats["semantic_hit"] += 1
            return best, "semantic", best_sim
        if best is not None:
            self.stats["rejected"] += 1
        self.stats["miss"] += 1
        return None, "miss", best_sim

    def put(self, tenant, model, params, prompt, response):
        exp = self.clock() + self.ttl
        k = self.key(tenant, model, params, prompt)
        self.exact[k] = (exp, response)
        self.exact.move_to_end(k)
        while len(self.exact) > self.capacity:
            self.exact.popitem(last=False)
        self.semantic.append((f"{tenant}|{model}|{k}", embed(prompt), exp, response))

    def hit_rate(self):
        s = self.stats
        served = s["exact_hit"] + s["semantic_hit"]
        total = served + s["miss"]
        return served / total if total else 0.0
```

The driver runs both against a fake clock, so the output is deterministic.
It admits five jobs for two tenants, advances the clock thirty seconds to show refill, settles one request whose real cost was ten times its estimate, and then puts six queries through the cache:

```python
lim = Limiter(tpm=6000, rpm=60, max_concurrent=2, clock=tick)
for tenant, est in [("acme", 500), ("acme", 500), ("acme", 4000),
                    ("acme", 2000), ("solo", 2000)]:
    d = lim.admit(tenant, est)
    if d.allowed:
        lim.settle(tenant, est, actual_tokens=est)

cache = TwoTierCache(threshold=0.85, ttl=60.0, clock=tick)
answer("acme", "How do I reset my password?")
answer("acme", "how do I RESET my password?")                    # exact, normalised
answer("acme", "How can I reset my password?")                   # true paraphrase
answer("acme", "What is the refund window for annual plans?")
answer("acme", "What is the refund window for monthly plans?")   # near miss
answer("globex", "How do I reset my password?")                  # other tenant
```

Running it, with the print statements restored:

```
====================================================================
1. TOKEN-METERED RATE LIMITING
====================================================================
  acme  est=  500 tok -> ALLOW                              remaining=5500
  acme  est=  500 tok -> ALLOW                              remaining=5000
  acme  est= 4000 tok -> ALLOW                              remaining=1000
  acme  est= 2000 tok -> DENY  (429, reason=tokens)         remaining=1000
        RateLimit-Policy: "tokens-per-min";q=6000;qu="tokens";w=60
        RateLimit: "tokens-per-min";r=1000;t=60
        Retry-After: 10
  solo  est= 2000 tok -> ALLOW                              remaining=4000
  acme  after 30s of refill -> ALLOW remaining=2000

  reconciling an underestimate:
    admitted on a 500-token estimate, remaining=5500
    generation actually used 5000; next request remaining=500 allowed=True

====================================================================
2. TWO-TIER CACHE (exact -> semantic)
====================================================================
  [miss     sim=0.000] acme: 'How do I reset my password?' -> upstream
  [exact    sim=1.000] acme: 'how do I RESET my password?'
  [semantic sim=0.866] acme: 'How can I reset my password?'
  [miss     sim=0.000] acme: 'What is the refund window for annual plans?' -> upstream
  [miss     sim=0.833] acme: 'What is the refund window for monthly plans?' -> upstream
  [miss     sim=0.000] globex: 'How do I reset my password?' -> upstream

  exact=1 semantic=1 miss=4 threshold-rejected=1
  hit rate = 33.3%   upstream calls = 4/6

====================================================================
3. WHERE THE THRESHOLD ACTUALLY SITS
====================================================================
  same question, reworded      -> 0.866  (want a HIT)
  one word changes the answer  -> 0.833  (want a MISS)
  usable margin between them   -> 0.033

  threshold 0.60: paraphrase=hit    annual/monthly=SERVES THE WRONG ANSWER
  threshold 0.75: paraphrase=hit    annual/monthly=SERVES THE WRONG ANSWER
  threshold 0.85: paraphrase=hit    annual/monthly=correctly rejected
  threshold 0.95: paraphrase=miss   annual/monthly=correctly rejected

====================================================================
4. TTL EXPIRY
====================================================================
  after 61s with ttl=60s -> miss (entry gone: True)
```

Read section 1 carefully.

The third request costs 4,000 tokens and is admitted, leaving 1,000.
The fourth wants 2,000 and is denied — not because the caller made too many requests (four is nothing against a limit of sixty) but because they asked for too much *work*.
`Retry-After: 10` is computed, not guessed: the bucket needs 1,000 more tokens and refills at 100 per second.

The `solo` tenant is unaffected, which is the whole point.

Then the reconciliation. A request admitted on a 500-token estimate that actually generates 5,000 does not get truncated — it completes, and the overspend is charged afterwards, so the *next* request sees a bucket at 500 instead of 5,500.
The caller pays for what they used, one request late.

Section 3 is the important one, and it is the reason this example exists.

Two pairs of questions.
One is a genuine paraphrase and should hit: 0.866.
One changes a single word in a way that completely changes the correct answer and must miss: 0.833.

**The gap is 0.033.**

At a threshold of 0.75 — which sounds cautious, and which you will find recommended in blog posts — the cache confidently tells a customer the wrong refund policy.
At 0.95 it stops working at all.
The window where it does the right thing on both is narrow, and it is narrow *here*, on a toy embedder, on two hand-picked pairs.

On real traffic the distributions overlap, which is exactly the vCache finding: there may be no threshold that gets every case right.
Which is why the guidance above is to derive the number from labelled data, log every hit with its score, and know your false-positive budget.

Anyone who tells you semantic caching is free has not measured the third section.

---

## Production checklist

**Rate limiting**

- [ ] Meter on tokens, not requests — input and output counted separately
- [ ] Separate cap on concurrent in-flight requests per caller
- [ ] Dollar-based limits if you route across models with different prices
- [ ] Two-phase charge: reserve an estimate on admission, reconcile on completion
- [ ] Limits at key, user, org, and global scope, evaluated cheapest-first
- [ ] Burst allowance (\\( B \\)) tuned separately from sustained rate (\\( r \\))
- [ ] New accounts start lower and graduate with age
- [ ] 429 with a computed `Retry-After`, never a hardcoded one
- [ ] `RateLimit` / `RateLimit-Policy` headers on success *and* failure, `X-RateLimit-*` alongside for compatibility
- [ ] Quota exhaustion and capacity exhaustion are distinguishable in the response body
- [ ] Shared state in Redis with the read-decide-write in a single Lua script
- [ ] Timestamps passed in as arguments; all keys declared in `KEYS`; hash tags for multi-key limits
- [ ] Fail-open vs fail-closed decided per limiter and controlled by a flag
- [ ] Local pre-filter in front of the global limiter to shed floods for free

**Queuing**

- [ ] Queue is bounded, with depth derived from Little's Law and your latency target
- [ ] Requests dropped when their queue age exceeds the client timeout
- [ ] Priority classes with a starvation guarantee for lower classes
- [ ] Load shedding by cost and by tier, applied at the edge
- [ ] A written degradation ladder — smaller model, fewer stages, stale cache — behind flags that have been tested
- [ ] Queue depth alerted on and exported to the autoscaler

**Caching**

- [ ] Exact-match tier before any semantic tier
- [ ] Cache key includes model+version, all sampling params, system prompt, tool defs, and prompt template version
- [ ] **Cache key includes tenant, and includes user whenever the response is personalized**
- [ ] Key derived from the same context object that built the prompt, not assembled by hand
- [ ] Semantic threshold chosen from labelled pairs, with the overlap plot saved somewhere
- [ ] Semantic cache disabled on personalized, financial, medical, and legal surfaces
- [ ] Vector index partitioned by tenant
- [ ] Every semantic hit logged with its similarity score
- [ ] A stated false-positive budget, measured against
- [ ] Embeddings cached by content hash, keyed by embedding model version
- [ ] Prompts structured prefix-stable so provider prompt caching engages; cache-read and cache-write token counts monitored
- [ ] TTLs set from data volatility, jittered ±10%
- [ ] Invalidation by version prefix, never by key scanning
- [ ] Single-flight coalescing on misses
- [ ] Stampede protection beyond that — probabilistic early expiry or stale-while-revalidate
- [ ] Hit rate tracked per tier, with alerts on sudden drops
- [ ] Cost avoided tracked in currency, for the budget conversation

---

## What an interviewer will probe

Two of the four canonical AI-engineering system design questions are this chapter.
These come up in some form nearly every time.

**"How would you rate limit an LLM API for millions of users?"**

Lead with the unit. Requests are the wrong unit because request cost varies by four orders of magnitude, so you meter tokens, concurrency, and dollars. Token bucket, because it handles variable cost naturally and gives you burst tolerance as a separate knob. State in Redis with the whole read-decide-write in one Lua script. Local pre-filter in each replica to shed obvious floods without a network hop. Limits at key, user, org, and global scope. 429 with a computed `Retry-After` and limit headers on every response.

The follow-up is always the two-phase charge, so get there before they ask.

**"You can't know the token cost until after you serve the request. So what do you charge?"**

Reserve an estimate on admission — `max_tokens`, or better, a rolling p90 of that caller's history. Reconcile on completion: charge the overspend, refund the underspend. Never truncate a stream to enforce a limit; let the overshoot land and make the next request pay. Say out loud that reserving at `max_tokens` without reconciling is why some APIs feel much stingier than their published numbers.

**"Why isn't a fixed window good enough?"**

The boundary burst: 100 at 11:59:59 and 100 at 12:00:00 is 200 in two seconds, both windows legal. Then say when you would accept it anyway — a coarse abuse filter in front of something with real headroom — and when you would not, which is anything in front of a GPU fleet sized at 20% margin.

**"You have twenty replicas. Where does the limiter state live?"**

Not in-process, because twenty replicas each enforcing the limit means 20× the limit gets through, and dividing by twenty is worse because traffic is not evenly distributed. Shared store, atomic update. Then the tradeoff: a Redis round trip is nothing against a three-second generation, so accuracy is cheap here — but a two-tier local/global setup still buys you flood protection at zero cost and reduces load on the limiter itself.

**"What breaks if you use `INCR` then `EXPIRE`?"**

Each is atomic; the pair is not. Crash between them and you have a counter with no TTL that never resets, permanently limiting that caller. More generally, any read-decide-write split across commands has a TOCTOU race, and under load the race is the common case. One Lua script.

**"How would you cache LLM responses efficiently?"**

Layers. Exact match first, keyed on normalized prompt plus model version plus sampling params plus system prompt plus template version plus tenant. Semantic second, with the risk stated. Embedding cache underneath both because embeddings are model calls too. And provider-side prompt caching, which is a different mechanism entirely — it caches KV state for a repeated prefix, not responses — and helps every request that misses your cache.

**"What's the risk with semantic caching?"**

That it returns a confidently wrong answer to a question nobody asked. Give the concrete example: annual versus monthly refund policy, near-identical embeddings, opposite answers. Cite that the similarity distributions of correct and incorrect hits overlap, so a single static threshold cannot cleanly separate them. Then the mitigations: threshold from labelled data, tenant partitioning, per-surface enablement, logging every hit's score, a cheap verification model on borderline hits, and a stated false-positive budget.

**"What must be in the cache key?"**

Everything that changes the answer — and then the one that gets missed: the user or tenant, whenever responses are personalized. Name it as a cross-user data leak rather than a caching bug, and note why it survives testing: one test user, low traffic, no error raised. The structural fix is to derive the key from the same context object that produced the prompt.

**"A hot cache entry expires and a hundred requests miss at once. What happens?"**

Cache stampede. A hundred identical model calls at the moment the cache stopped protecting you, so the cache has amplified load rather than reduced it. Single-flight coalescing is the primary fix. Then probabilistic early expiration so one request refreshes early while others still get the cached value, stale-while-revalidate if slightly stale is acceptable, jittered TTLs so entries do not expire in lockstep, and pre-warming for known-hot keys.

**"You're at capacity. Queue or reject?"**

Both, with a boundary. Queue briefly, bounded by Little's Law against your latency target, and drop anything that has waited past the client timeout. An unbounded queue is a slower failure: clients time out and retry, so you burn GPU on answers nobody reads. Then priority classes, shedding by cost first because one long request costs ten short ones, and degradation — a smaller model or a stale cache entry beats a 429.

**"What hit rate should we expect?"**

Refuse the universal number. 5–15% for an open-ended assistant, 30–60% for a support bot with a fat head, higher for internal templated workloads. Then pivot to the diagnostics, which is what they are really testing: near-zero means an over-specific key with something varying in it; suspiciously high on personalized content means an under-specific key and you should go check for a leak today; a sudden drop means a model version, prompt template, or context field changed the key shape.

**"Gateway or application?"**

Both, split by what each can know. Coarse limits at the gateway — requests per second, connections, body size, IP — because rejecting at the edge is cheapest and Envoy or Kong have already solved it. Token-aware limits in your service, because only your service can tokenize the input, estimate the output, and reconcile afterwards. Then say plainly that you would not contort a gateway into doing token accounting, and you would not skip the gateway because it cannot.

---

## Further reading

**Rate limiting algorithms**

- Brandur Leach, *Rate Limiting, Cells, and GCRA* — the canonical GCRA explanation, including why removing the drip process removes a class of failure. https://brandur.org/rate-limiting
- Wikipedia, *Generic cell rate algorithm* — the formal definition from ATM networking. https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm
- Redis, *Build 5 Rate Limiters with Redis* — fixed window, sliding log, sliding counter, token bucket, with the Lua scripts and an explicit treatment of why atomicity is required. https://redis.io/tutorials/howtos/ratelimiting/
- Cloudflare, *How we built rate limiting capable of scaling to millions of domains* — the sliding window counter approximation at scale. https://blog.cloudflare.com/counting-things-a-lot-of-different-things/

**Standards and headers**

- IETF HTTPAPI WG, *RateLimit header fields for HTTP* (draft-ietf-httpapi-ratelimit-headers-11, May 2026) — `RateLimit` and `RateLimit-Policy`, including the `qu` quota-unit parameter. https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/
- The working group's repository, with examples and discussion. https://github.com/ietf-wg-httpapi/ratelimit-headers

**Gateways**

- Envoy, *Global rate limiting* — the rate limit service architecture and how local and global limiting compose. https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting
- Envoy, *Local rate limit filter* — the in-process token bucket. https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/local_rate_limit_filter
- NGINX, *Rate Limiting with NGINX* — leaky bucket, and what `burst` and `nodelay` actually do. https://blog.nginx.org/blog/rate-limiting-nginx

**Semantic caching**

- Zilliz, *GPTCache* — the reference open-source semantic cache: embedder, vector store, similarity evaluator, eviction. https://github.com/zilliztech/GPTCache
- *GPTCache: An Open-Source Semantic Cache for LLM Applications* (paper). https://openreview.net/pdf?id=ivwM8NwM4Z
- *vCache: Verified Semantic Prompt Caching* (arXiv 2502.03771) — why static thresholds fail, with the overlapping-distribution analysis and a per-embedding adaptive alternative. https://arxiv.org/abs/2502.03771

**Provider prompt caching**

- Anthropic, *Prompt caching* — `cache_control`, breakpoints, minimum token counts, 5m and 1h TTLs, and the 1.25×/2×/0.1× pricing multipliers. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- OpenAI, *Prompt caching* — automatic caching above 1,024 tokens, `prompt_cache_key`, and prefix structuring. https://developers.openai.com/api/docs/guides/prompt-caching
- Amazon Bedrock, *Prompt caching* — the same idea across Bedrock-hosted models. https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html

**Stampede**

- Vattani, Chierichetti & Lowenstein, *Optimal Probabilistic Cache Stampede Prevention* (VLDB 2015) — the XFetch algorithm. https://cseweb.ucsd.edu/~avattani/papers/cache_stampede.pdf
- Go's `singleflight` package — the canonical request-coalescing implementation. https://pkg.go.dev/golang.org/x/sync/singleflight

**Sibling repositories**

- `llm-serving-inference-guide` — this guide's own autoscaling chapters for what happens when capacity is the real constraint, and its monitoring chapters for the queue-depth and cache-hit signals this topic assumes you can see.
- `learn-production-agent` — the agent-layer view of the same economics: budgets charged before spending, model routing, and cost per *successful task* rather than per call. That book governs an agent you control; this topic enforces limits against a caller you do not.
