# Hardware Roots of Trust: Enclave Attestation and Verifiable Credentials for Agent Identity

In November 2024, a cloud infrastructure breach exposed a pattern that security teams had warned about for years. An autonomous deployment agent running on a shared Kubernetes cluster had its process memory dumped after an attacker exploited a privilege escalation vulnerability on the host node. 

The attacker extracted the agent API key and database credentials. Over the next forty-eight hours, the attacker executed infrastructure modifications across three cloud regions. Every cloud audit log attributed the actions to the deployment bot. The credentials were legitimate, the API tokens were valid, and the IAM policies were correctly mapped.

Every authentication check passed. The incident exposed a deeper structural limitation: software-based identity cannot prove execution integrity.

A traditional API key or service account proves only that someone possesses a string. It proves nothing about what software generated the request, whether the system prompt was modified, whether the container binary was patched, or whether the execution environment was compromised.

As agents transition from conversational chatbots to autonomous economic actors executing transactions and managing infrastructure, identity based on pre-shared secrets becomes untenable. 

The future of agent identity rests on two converging foundations: hardware-backed confidential computing that proves execution integrity through remote attestation, and portable Verifiable Credentials that decouple identity from proprietary cloud providers.

```
Traditional Agent Identity (Ambient Secret):
[Host Machine / Hypervisor] (Can dump RAM)
       │
 [Agent Process] ──(Holds Static API Key in Env Var)──> [Cloud IAM / API]
       │
Attacker Dumps Memory -> Extracts Key -> Full Impersonation Anywhere

Hardware-Attested Agent Identity:
[Trusted Execution Environment (AMD SEV-SNP / Intel TDX)]
       │
 [Agent Process]
 ├── Hardware AES Memory Encryption (Host Hypervisor Cannot Read)
 ├── Silicon Root of Trust Measures Code + Prompt Hashes
 └── Signs Remote Attestation Report
       │
       ▼
 [Key Broker Service (KBS)] ──(Validates Silicon Signature & Hash)──> Mints Ephemeral Verifiable Credential
       │
 [Agent Presents Credential to External API / Peer Agent]
 Verification is Cryptographic, Portable, and Hardware-Proven
```

---

## The Secret Zero Dilemma

Every software identity system faces the "Secret Zero" problem: to retrieve a secret securely from a vault, an application must already possess an initial secret proving who it is.

In cloud systems, teams bootstrap Secret Zero using container metadata, instance profile metadata, or environment variables. This creates a brittle security boundary. If the host operating system, the container runtime, or a sidecar process is compromised, the attacker reads Secret Zero and assumes the identity of the agent.

For autonomous agents, Secret Zero is unusually dangerous:
1. **Agents are non-deterministic:** Unlike traditional microservices that execute fixed code paths, an agent plans its trajectory dynamically. A hijacked agent can execute novel sequences of destructive commands that static firewall rules fail to anticipate.
2. **Prompts are code:** In an agent, the system prompt defines business rules, constraints, and safety invariants. If an attacker modifies the system prompt on disk before the container boots, the agent code remains untouched, but its behavior changes completely. Static binary hashes cannot detect this tampering.
3. **Cross-boundary coordination:** Agents increasingly interact with external APIs and other autonomous agents across corporate firewalls. Relying on Amazon Web Services IAM or Google Cloud Service Accounts forces external partners to accept proprietary cloud identities that cannot be audited outside the host environment.

Solving Secret Zero requires rooting identity directly in silicon.

---

## Hardware-Enforced Identity: Confidential Computing and Attestation

Confidential computing protects data while it is being processed in memory. Using hardware-isolated Trusted Execution Environments (TEEs) such as AMD SEV-SNP (Secure Encrypted Virtualization-Secure Nested Paging), Intel TDX (Trust Domain Extensions), or AWS Nitro Enclaves, the CPU isolates the agent execution context entirely.

Even if an attacker gains root access to the host operating system or hypervisor, the CPU hardware memory controller encrypts all memory pages using ephemeral AES keys generated inside the processor. The host cannot inspect or tamper with agent memory.

```
+---------------------------------------------------------------+
| Untrusted Host / Hypervisor (Zero Access to Enclave Memory)   |
|                                                               |
|   +-------------------------------------------------------+   |
|   | Hardware-Isolated Trusted Execution Environment (TEE) |   |
|   |                                                       |   |
|   |  Agent Runtime Binary (SHA-256: 8a91...)              |   |
|   |  System Prompt & Tools (SHA-256: 4f12...)             |   |
|   |  In-Memory Private Key (Ed25519)                      |   |
|   |                                                       |   |
|   |  Hardware Measurement Engine                          |   |
|   |  (Computes cryptographic hash of all components)      |   |
|   +-------------------------------------------------------+   |
|                               |                               |
+-------------------------------+-------------------------------+
                                │
                        Requests Quote
                                │
                                ▼
        [CPU Silicon Security Processor (AMD / Intel)]
        Signs Attestation Report with Factory Endorsement Key
```

### The Remote Attestation Mechanism
When an agent boots inside a TEE, it does not hold an API key. It holds only the hardware measurement of its state.

The hardware security processor measures every layer of the running agent:
- Measurement Register 0 (MR0): Bootloader and firmware state.
- Measurement Register 1 (MR1): OS kernel and initrd hashes.
- Measurement Register 2 (MR2): Agent container filesystem and binary digest.
- Measurement Register 3 (MR3): Initial runtime configuration, including the SHA-256 hash of the system prompt and tool definitions.

The agent requests an attestation quote from the processor. The processor compiles the measurement registers into a structured report, appends an ephemeral public key generated by the agent, and signs the payload with its hardware Attestation Key. 

The root certificate for this key traces directly back to the silicon manufacturer (AMD, Intel, or Amazon). No software on the host can forge this signature.

### Bootstrapping Identity via the Key Broker Service
With the attestation report in hand, the agent contacts an external Key Broker Service (KBS) to bootstrap its credentials:

```python
import base64
import hashlib
from typing import Any
import httpx
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

class EnclaveAgentBootstrapper:
    def __init__(self, kbs_endpoint: str, golden_measurement: str):
        self.kbs_endpoint = kbs_endpoint
        self.golden_measurement = golden_measurement
        # Generate ephemeral keypair inside enclave memory
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def get_public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    def retrieve_hardware_attestation_quote(self, user_data: bytes) -> bytes:
        # In a real TEE (e.g. /dev/sev-guest), write user_data (public key hash) 
        # to the hardware driver and read the signed attestation report.
        # This guarantees the public key is cryptographically bound to the report.
        driver_path = "/dev/sev-guest"
        try:
            with open(driver_path, "rb") as dev:
                # Issue ioctl request to retrieve hardware quote
                pass
        except FileNotFoundError:
            # Fallback for mock testing in development
            pass
        
        # Return mock hardware attestation structure for architecture demonstration
        return b"HARDWARE_SIGNED_ATTESTATION_REPORT"

    def bootstrap_identity(self) -> dict[str, Any]:
        pub_bytes = self.get_public_bytes()
        # Bind ephemeral key hash to the attestation quote
        key_hash = hashlib.sha256(pub_bytes).digest()
        quote = self.retrieve_hardware_attestation_quote(user_data=key_hash)

        payload = {
            "attestation_quote": base64.b64encode(quote).decode("utf-8"),
            "public_key": base64.b64encode(pub_bytes).decode("utf-8")
        }

        # Send to Key Broker Service
        response = httpx.post(f"{self.kbs_endpoint}/v1/attest", json=payload)
        if response.status_code != 200:
            raise PermissionError(f"Attestation failed: {response.text}")

        # The KBS releases the dynamic task credential bound to our verified enclave
        return response.json()
```

The Key Broker Service validates the hardware certificate against the manufacturer PKI. It then compares the measurement hash in the report against a known-good reference value (the "golden measurement"). 

If someone tampered with the system prompt on disk, MR3 changes. The hash check fails. The Key Broker Service drops the connection, releasing zero secrets.

---

## Portable Identity: W3C Verifiable Credentials and DIDs

Hardware attestation proves integrity. It does not provide a standard format for presenting identity across distributed enterprise boundaries.

If an agent built by Company A needs to negotiate purchase orders with an agent operated by Company B, Company B cannot query Company A's internal Key Broker Service or cloud directory.

To make agent identity portable, systems use W3C Decentralized Identifiers (DIDs) and Verifiable Credentials (VCs).

Instead of an AWS IAM ARN, the agent identifies itself using a DID, such as `did:key` or `did:tdx`. The Key Broker Service acts as an issuer, converting the raw hardware attestation report into a signed Verifiable Credential:

```json
{
  "@context": [
    "https://www.w3.org/2018/credentials/v1",
    "https://schema.org"
  ],
  "id": "urn:uuid:89a104f2-4912-4c20-a891-b92819401823",
  "type": ["VerifiableCredential", "AgentWorkloadCredential"],
  "issuer": "did:web:auth.enterprise-corp.com",
  "issuanceDate": "2026-09-28T05:30:00Z",
  "expirationDate": "2026-09-28T06:30:00Z",
  "credentialSubject": {
    "id": "did:key:z6MkuV4K9Vn2HjC6s8h7q5V8P9k0L1M2N3O4P5Q6R7S8T9U",
    "operator": "Acme Logistics Inc",
    "enclave_platform": "AMD-SEV-SNP",
    "binary_digest": "sha256:d891e4f9b2049182309182049182309182049182309182049182309182049182",
    "prompt_digest": "sha256:4a08bc819e018239018203918203918203918203918203918203918203918203",
    "compliance_tier": "SOC2_TYPE_II_AUDITED",
    "allowed_capabilities": [
      "procurement.rfq.submit",
      "procurement.order.sign"
    ],
    "spend_limit_usd": 10000.00
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "created": "2026-09-28T05:30:05Z",
    "verificationMethod": "did:web:auth.enterprise-corp.com#key-1",
    "proofPurpose": "assertionMethod",
    "proofValue": "z3h8d92kd...8192kds="
  }
}
```

This credential carries high technical value for multi-agent coordination:
1. **Self-contained proof:** The credential contains the exact software hash and prompt hash running in the enclave.
2. **Cryptographic binding:** The subject ID is the public key generated inside the agent's memory. Any request presented with this credential must be signed by the corresponding private key using a proof-of-possession mechanism.
3. **Vendor neutrality:** Company B can verify the signature using standard public cryptography without needing network access to Company A's internal identity infrastructure.

---

## Agent-to-Agent Mutual Attestation Protocol

When two autonomous agents collaborate across organizational boundaries, they must establish trust without human intermediaries.

The protocol follows an explicit handshake:

```
[Agent A (Buyer)]                                      [Agent B (Supplier)]
       │                                                        │
       │ 1. Connect via mTLS                                    │
       │ ─────────────────────────────────────────────────────> │
       │                                                        │
       │ 2. Present Verifiable Credential A                     │
       │    (Signed by Buyer KBS + AMD Attestation)             │
       │ ─────────────────────────────────────────────────────> │
       │                                                        │
       │ 3. Present Verifiable Credential B                     │
       │    (Signed by Supplier KBS + Intel TDX Attestation)    │
       │ <───────────────────────────────────────────────────── │
       │                                                        │
       │ 4. Mutual Verification of Measurements & Policy Bounds │
       │    Both agents confirm:                                │
       │    - Peer runs in certified TEE                        │
       │    - Peer software matches audited digest              │
       │    - Counterparty capabilities cover contract terms    │
       │                                                        │
       │ 5. Execute Encrypted Negotiation Session               │
       │ <════════════════════════════════════════════════════> │
```

During step 4, each agent evaluates the counterparty's credential against local policy:
- Does the supplier agent run in an audited hardware enclave?
- Is the software hash recognized on the industry registry of verified procurement systems?
- Is the spend limit authorized by the issuer?

If either agent detects an untrusted container hash or an invalid hardware signature, the socket terminates before any commercial terms or proprietary data are exchanged.

---

## The Identity Paradigm Shift

The transition from pre-shared secrets to hardware-attested verifiable credentials marks a structural evolution in infrastructure architecture:

| Dimension | Legacy Service Accounts | Hardware-Attested Verifiable Identity |
| :--- | :--- | :--- |
| **Trust Anchor** | Software database (Cloud IAM registry) | Silicon processor keys (AMD, Intel, AWS Nitro) |
| **Integrity Scope** | Proves possession of a secret key | Proves code, system prompt, and memory integrity |
| **Theft Resistance** | High risk; tokens can be extracted from RAM | Zero extraction; memory encrypted by hardware AES |
| **Portability** | Locked to proprietary cloud providers | Standardized via W3C DIDs and Verifiable Credentials |
| **Secret Zero** | Permanent secrets stored in disk or env vars | Eliminated; credentials minted only after attestation |
| **Federation** | Requires complex cross-account IAM federation | Native cryptographic verification over public keys |

Securing the next decade of autonomous software requires discarding the fiction that an API key proves identity. 

In a world where agents execute complex decisions at wire speed, identity must be rooted in physical silicon, mathematically attested, and verified at the boundary of every execution.
