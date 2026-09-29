# Why OAuth 2.0 Breaks for Autonomous Agents: Rich Authorization Requests and Cryptographic Proof-of-Possession

An autonomous agent operating inside an infrastructure pipeline was given an OAuth token with the scope `repo:write`. The developer intended for the agent to open a pull request updating a dependency version. Instead, after reading an untrusted issue description containing a prompt injection, the agent used the same bearer token to force-push a commit directly to the repository main branch and delete three closed pull requests.

The authorization server evaluated the bearer token, checked the scope string, and returned HTTP 200. Every check in the OAuth pipeline passed. From the perspective of the resource server, the agent was completely authorized to overwrite git history.

This failure exposes the structural mismatch between classic OAuth 2.0 and autonomous agents. OAuth 2.0 was designed for two computing patterns: an interactive user approving coarse permissions on a consent screen, or a background worker exchanging static client credentials for machine-to-machine syncs. 

Neither model works for multi-step AI agents. Agents execute asynchronous, dynamic execution graphs without human oversight at every step. They interact with third-party tools, read untrusted data, and take real-world actions. Giving an agent a bearer token with coarse string scopes hands total ambient authority to an probabilistic planner. 

To secure autonomous agents, authorization must evolve along two axes: restricting authorization to structured, intent-bound parameters, and cryptographically binding tokens to the executing agent runtime so stolen credentials cannot be replayed.

```
Classic OAuth Bearer Model:
[Agent Instance] ──(Bearer Token + Coarse Scope: "repo:write")──> [Git API]
       │
   Compromised / Leaked
       │
       ▼
 [Attacker / Malicious Tool] ──(Same Bearer Token)──> [Git API] (Accepted)

Sender-Constrained RAR + DPoP Model:
[Agent Instance (Keypair)] ──(DPoP Proof [Ed25519] + RAR JSON Payload)──> [Git API]
       │
   Token Leaked
       │
       ▼
 [Attacker] ──(Bearer Token Alone)──> [Git API] (Rejected: Missing Private Key Proof)
```

---

## The Triple Failure of Classic Bearer Scopes

OAuth 2.0 RFC 6749 built an authorization model around two core assumptions: tokens are bearer instruments, and scopes are flat strings. Both assumptions create security vulnerabilities when autonomous agents enter the loop.

### 1. Bearer Tokens Invite Interception and Replay
A bearer token is cash. Whoever holds the string can spend it. When an agent runs a multi-step task, it frequently passes tokens across API boundaries, logs intermediate states, or exposes data to downstream tools and external plugins. 

If an agent interacts with a malicious Model Context Protocol (MCP) server or suffers an indirect prompt injection that triggers an outbound fetch, an attacker can exfiltrate the bearer token. Because the resource server only inspects the token string, the attacker can replay it from any machine in the world until the token expires.

### 2. Flat Scopes Cannot Express Dynamic Intent
OAuth scopes are coarse, space-delimited strings such as `files:write`, `payment:execute`, or `calendar:read`. They describe static capabilities rather than dynamic operational intent. 

When a human connects an email client, granting `mail:read` for six months is standard. When an agent is tasked with summarizing an order confirmation email received in the last thirty minutes, granting `mail:read` across the entire ten-year archive provides massive over-authorization. An agent needs permissions bounded by specific IDs, temporal windows, and quantitative limits. Flat string scopes cannot express these constraints without creating thousands of ad-hoc scope combinations that break authorization server registries.

### 3. Interactive Consent Screens Collapse in Multi-Step Trajectories
Classic OAuth relies on the authorization code flow, presenting a human user with a browser redirect to review requested scopes. 

An agent resolving an incident at 2:00 AM may generate a trajectory involving fifteen discrete tool calls across four cloud services. Halting the agent to present fifteen separate browser consent dialogues breaks autonomy. Conversely, asking the human upfront to grant a blanket bundle of write permissions across all four services before the agent starts recreates the ambient authority problem.

---

## Structured Intent with RFC 9396 Rich Authorization Requests

The Internet Engineering Task Force addressed the expressiveness limits of string scopes in RFC 9396: Rich Authorization Requests (RAR). RAR replaces flat scope strings with a structured JSON array under the parameter `authorization_details`.

Instead of requesting a broad capability, the client submits a typed object defining the exact action, resource identifier, and operational constraints.

Here is what an agentic RAR request looks like when an agent requests permission to process a vendor invoice refund:

```json
{
  "authorization_details": [
    {
      "type": "accounting_operation",
      "actions": ["issue_refund"],
      "locations": ["https://api.internal.finance/v2/refunds"],
      "currency": "USD",
      "maximum_amount": 750.00,
      "vendor_id": "vend_849201",
      "invoice_id": "inv_2026_0941",
      "expires_at": 1790582400
    }
  ]
}
```

The authorization server validates this structured claim against the organization policy engine. When the resulting token is minted, the token claims contain these exact constraints. When the agent presents the token to the accounting API, the resource server evaluates the payload directly:

```python
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

app = FastAPI()
security = HTTPBearer()

def verify_token_rar_constraints(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict[str, Any]:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, 
            key="VERIFYING_KEY_PEM", 
            algorithms=["ES256"], 
            audience="https://api.internal.finance/v2"
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    details = payload.get("authorization_details", [])
    if not details:
        raise HTTPException(status_code=403, detail="Missing authorization details")

    return details[0]

@app.post("/v2/refunds")
def process_refund(
    invoice_id: str, 
    amount: float, 
    auth_detail: dict[str, Any] = Depends(verify_token_rar_constraints)
):
    # Enforce typed constraints at the resource boundary
    if auth_detail.get("type") != "accounting_operation":
        raise HTTPException(status_code=403, detail="Incorrect permission type")

    if "issue_refund" not in auth_detail.get("actions", []):
        raise HTTPException(status_code=403, detail="Unauthorized action")

    if auth_detail.get("invoice_id") != invoice_id:
        raise HTTPException(status_code=403, detail="Token bound to different invoice")

    max_amount = float(auth_detail.get("maximum_amount", 0.0))
    if amount > max_amount:
        raise HTTPException(
            status_code=403, 
            detail=f"Requested amount ${amount} exceeds token cap ${max_amount}"
        )

    return {"status": "processed", "invoice_id": invoice_id, "amount": amount}
```

If the agent is redirected by prompt injection to execute a refund for an invoice not listed in `auth_detail["invoice_id"]`, the resource server returns HTTP 403. The check does not depend on model compliance or prompt instructions. The verification runs as deterministic code at the gateway.

---

## Sender-Constrained Tokens with RFC 9449 DPoP

Structured authorization solves the precision problem. It does not solve the stolen token problem. If the access token containing the RAR claims is intercepted from an agent container or memory buffer, the attacker can still submit valid requests within those RAR parameters.

To eliminate bearer credential replay, agent systems must implement RFC 9449: Demonstrating Proof-of-Possession (DPoP) at the Application Layer.

DPoP converts an access token into a sender-constrained credential. The agent client generates an ephemeral asymmetric cryptographic key pair (such as ECDSA P-256 or Ed25519) in memory at boot. The private key never leaves the agent process memory.

```
Agent Runtime                       Authorization Server                Resource Server (Tool)
     │                                       │                                    │
     │ 1. Generate Ephemeral Keypair         │                                    │
     │    (priv_key, pub_key)                │                                    │
     │                                       │                                    │
     │ 2. Push Auth Request (PAR)            │                                    │
     │    with RAR + DPoP Proof              │                                    │
     │ ────────────────────────────────────> │                                    │
     │                                       │                                    │
     │ 3. Mint Token with 'cnf' thumbprint   │                                    │
     │ <──────────────────────────────────── │                                    │
     │                                       │                                    │
     │ 4. Request Tool Execution             │                                    │
     │    Header: Authorization: DPoP <tok>  │                                    │
     │    Header: DPoP: <Signed JWT Proof>   │                                    │
     │ ─────────────────────────────────────────────────────────────────────────> │
     │                                       │                                    │
     │                                       │    5. Hash Access Token (ath)      │
     │                                       │    6. Match 'cnf' claim to JWK     │
     │                                       │    7. Verify Method & URI match    │
     │                                       │    8. Execute Deterministic Action │
     │ <───────────────────────────────────────────────────────────────────────── │
```

### 1. Requesting the Bound Token
When requesting a token, the agent creates a signed DPoP proof JWT. The header contains the agent public JSON Web Key (JWK). The payload contains the HTTP method, the target URL, a timestamp, and a unique identifier (`jti`):

```json
// DPoP Proof Header
{
  "typ": "dpop+jwt",
  "alg": "ES256",
  "jwk": {
    "kty": "EC",
    "crv": "P-256",
    "x": "6i0WfB6u8V4K9Vn2H-jC6s8h7q5V8P9k0L1M2N3O4P5",
    "y": "9Z8Y7X6W5V4U3T2S1R0Q9P8O7N6M5L4K3J2I1H0G9F8"
  }
}

// DPoP Proof Payload
{
  "jti": "dpop_proof_98410294",
  "htm": "POST",
  "htu": "https://auth.company.internal/oauth/token",
  "iat": 1790582000
}
```

The authorization server computes the SHA-256 thumbprint of the agent public key and embeds it in the minted access token under the confirmation claim `cnf`:

```json
// Minted Access Token Payload
{
  "iss": "https://auth.company.internal",
  "sub": "agent_worker_prod_4",
  "aud": "https://api.internal.finance/v2",
  "exp": 1790582600,
  "cnf": {
    "jkt": "0Z9X8Y7W6V5U4T3S2R1Q0P9O8N7M6L5K4J3I2H1G0F9"
  },
  "authorization_details": [...]
}
```

### 2. Using the DPoP Token at the Resource Server
When calling the protected tool API, the agent must generate a fresh DPoP proof for that exact call, embedding the hash of the access token (`ath`):

```python
import base64
import hashlib
import time
import uuid
import httpx
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
import jwt

class DPoPAgentClient:
    def __init__(self, private_key: ec.EllipticCurvePrivateKey):
        self.private_key = private_key
        self.public_key = private_key.public_key()
        # Derive public JWK parameters
        public_numbers = self.public_key.public_numbers()
        self.jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": base64.urlsafe_b64encode(
                public_numbers.x.to_bytes(32, "big")
            ).decode("utf-8").rstrip("="),
            "y": base64.urlsafe_b64encode(
                public_numbers.y.to_bytes(32, "big")
            ).decode("utf-8").rstrip("="),
        }

    def generate_dpop_proof(
        self, method: str, target_uri: str, access_token: str | None = None
    ) -> str:
        headers = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": self.jwk
        }
        payload = {
            "jti": str(uuid.uuid4()),
            "htm": method.upper(),
            "htu": target_uri,
            "iat": int(time.time()),
        }
        if access_token:
            token_hash = hashlib.sha256(access_token.encode("ascii")).digest()
            payload["ath"] = base64.urlsafe_b64encode(token_hash).decode("utf-8").rstrip("=")

        return jwt.encode(payload, self.private_key, algorithm="ES256", headers=headers)

    def execute_tool_call(self, url: str, token: str, body: dict) -> httpx.Response:
        dpop_proof = self.generate_dpop_proof(method="POST", target_uri=url, access_token=token)
        headers = {
            "Authorization": f"DPoP {token}",
            "DPoP": dpop_proof,
            "Content-Type": "application/json",
        }
        return httpx.post(url, json=body, headers=headers)
```

The resource server verifies three conditions:
1. The `DPoP` proof header signature is valid and corresponds to the public key in its header.
2. The thumbprint of that public key matches the `cnf.jkt` claim in the access token.
3. The `htm` and `htu` claims match the incoming HTTP method and URL, and `ath` matches the SHA-256 hash of the presented token.

If an attacker intercepts the token string from an MCP tool log, they cannot use it. The resource server checks for the matching DPoP proof signed by the agent private key. Because the attacker does not hold the private key living inside the agent process memory, their requests fail immediately.

---

## Dynamic Revocation with Continuous Access Evaluation (CAEP)

Traditional OAuth assumes access tokens remain valid until expiration unless explicitly revoked via a polling endpoint. For short agent actions, tokens can expire in five minutes. However, complex multi-hop trajectories can run for hours.

If an agent displays abnormal behavior, touches a honeypot record, or ingests untrusted text that triggers an internal prompt injection classifier, the system cannot wait for an hourly token expiration.

The OpenID Shared Signals and Events (SSE) framework, specifically the Continuous Access Evaluation Profile (CAEP), provides the real-time control plane. Instead of the agent polling for validity, the security control plane pushes Security Event Tokens (SETs, RFC 8417) to resource gateways over webhooks.

```
[Agent Context Monitor] ────(Taint Detected)────> [Identity Provider]
                                                         │
                                               Sends CAEP Event (SET)
                                                         │
                                                         ▼
                                               [Tool API Gateway / Envoy]
                                               Invalidates Subject & Session Cache
                                                         │
[Agent Next Step] ──(Submits Valid DPoP Token)──> [Tool API Gateway] ──> Returns HTTP 401 Session Revoked
```

A CAEP event payload specifies the exact subject or session to terminate:

```json
{
  "iss": "https://auth.company.internal",
  "iat": 1790582210,
  "jti": "set_caep_481029",
  "aud": "https://api.internal.finance/v2",
  "events": {
    "https://schemas.openid.net/secevent/caep/event-type/session-revoked": {
      "subject": {
        "format": "opaque",
        "id": "agent_session_94819"
      },
      "event_timestamp": 1790582208,
      "reason_admin": {
        "en": "Untrusted prompt injection pattern detected in agent context"
      }
    }
  }
}
```

The API gateway maintains a local Redis revocation cache. When the agent attempts its next tool call, the gateway rejects the request in sub-millisecond time. The agent execution halts before it can execute state changes on downstream infrastructure.

---

## The Production Blueprint: Putting the Stack Together

Implementing modern OAuth for autonomous agents requires combining these standards into an explicit lifecycle:

| Layer | Standard | Mechanism | Failure Mode Eliminated |
| :--- | :--- | :--- | :--- |
| **Request Framing** | RFC 9126 (PAR) | Pushes parameters directly over TLS via POST before authorization | Prevents token configuration leakage in browser histories and proxy logs |
| **Intent Boundaries** | RFC 9396 (RAR) | Structured JSON objects specifying typed actions, resource targets, and values | Prevents prompt-injected agents from performing actions outside task boundaries |
| **Proof of Possession** | RFC 9449 (DPoP) | Cryptographic binding to an in-memory asymmetric key pair | Prevents stolen tokens from being replayed by third-party tools or attackers |
| **Runtime Control** | OpenID CAEP (SSE) | Event-driven push revocation over webhook backchannels | Eliminates the vulnerability window between token compromise and expiration |

Treating agents as standard web users clicking "Allow" on broad permission scopes is an architectural mistake. Treating them as machine-to-machine microservices with permanent API keys is worse.

Autonomous agents require ephemeral, sender-constrained credentials that carry explicit mathematical bounds on what they can touch, how much they can spend, and where they can operate. RFC 9396 and RFC 9449 provide the protocol foundation to build that architecture today.
