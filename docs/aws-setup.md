# AWS configuration

This service uses Amazon Bedrock Runtime directly. Bedrock AgentCore is neither required nor
integrated.

## GLM-5 configuration

The analysis agents use Z.AI GLM-5 through the Bedrock Converse API:

```dotenv
AWS_REGION=us-west-2
BEDROCK_ANALYSIS_MODEL_ID=zai.glm-5
BEDROCK_ANALYSIS_CONTEXT_WINDOW_TOKENS=200000
BEDROCK_ANALYSIS_MAX_OUTPUT_TOKENS=128000
BEDROCK_CHAT_MODEL_ID=openai.gpt-oss-120b-1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_READ_TIMEOUT_SECONDS=3600
```

`zai.glm-5` is an in-region model ID. GLM-5 does not provide Geo or Global inference IDs, so do
not prefix it with `global.` and do not use an inference-profile ARN. AWS currently lists GLM-5
in `us-east-1`, `us-east-2`, `us-west-2`, `eu-north-1`, `eu-west-2`, `ap-northeast-1`,
`ap-south-1`, `ap-southeast-2`, `ap-southeast-3`, `ap-southeast-4`, and `sa-east-1`. The AWS
Region used by credentials and the application must be one of the regions where the model is
available.

GLM-5 has a 200K-token context window and a 128K-token maximum output. The application starts
requests at `agents.max_output_tokens` (32,768 by default), increases the limit only after a
`max_tokens` stop, and caps it by both the configured model limit and the estimated remaining
context. Source batches and discovery prompts are conservatively bounded. Knowledge generation
uses graph-grounded map/reduce chunks, so a large repository is not sent as one oversized prompt.

## GPT-OSS 120B architecture chat

Architecture chat uses the independently configurable Bedrock Runtime model ID
`openai.gpt-oss-120b-1:0`. Its 128K-token context window and 16K maximum output are represented by
the bounded `chat` settings; the default response limit is 4,096 tokens. The application does not
send entire repositories to chat. It reads the latest published reader editions, selects relevant
sections locally, and sends a bounded source-labelled prompt through the Converse API. Returned JSON
is validated locally and citations outside those supplied labels are rejected.

AWS also exposes GPT-OSS 120B through Bedrock Mantle as `openai.gpt-oss-120b`. This project continues
to use Bedrock Runtime because it already has a least-privilege IAM/Converse adapter and does not need
another SDK, API key, or AgentCore deployment.

## Why Bedrock Runtime instead of Bedrock Mantle

AWS supports GLM-5 on both endpoints:

- Bedrock Runtime: Converse and InvokeModel with native IAM attribution, request metadata,
  guardrails, and Converse structured output.
- Bedrock Mantle: OpenAI-compatible Chat Completions at
  `https://bedrock-mantle.{region}.api.aws/v1`, plus Mantle projects and workspaces.

This application deliberately keeps Bedrock Runtime because its agent adapter already uses
Converse, request metadata, optional Bedrock Guardrails, and `outputConfig.textFormat` JSON-schema
responses. Mantle would add an OpenAI client and API-key surface without improving this stateless
pipeline. Mantle can be added later as a separate adapter behind `StructuredModelClient`; no
domain or agent code needs to change.

## Structured output

`agents.native_structured_output` is enabled for GLM-5. Pydantic response schemas are converted to
the JSON Schema subset supported by Bedrock before being sent in `outputConfig.textFormat`.
Unsupported numerical and string constraints remain enforced locally after the response. Every
response is still parsed and validated locally, and invalid evidence fails closed.

The first request for a new schema can take a few minutes while Bedrock compiles it. AWS caches a
successfully compiled grammar for 24 hours. The application also stores validated specialist
responses in `artifacts/cache/agents`; the model ID is part of the cache key, so previous-model cache
entries cannot be reused for GLM-5.

Model-specific adaptive thinking fields are disabled for GLM-5. Sampling parameters are intentionally
omitted; prompts, schema constraints, deterministic evidence checks, and graph traversal provide
consistency.

## Enable model access

Bedrock foundation models are normally available automatically when the account has the required
AWS Marketplace permissions. For a first invocation of a third-party model, an AWS administrator
may need `aws-marketplace:Subscribe`, `aws-marketplace:Unsubscribe`, and
`aws-marketplace:ViewSubscriptions`, and the account must have a valid Marketplace payment method.
Activation can take a few minutes. Keep these subscription permissions on an onboarding/admin
role rather than the application's runtime role.

In the Amazon Bedrock console, choose the same Region as `AWS_REGION`, open **Model catalog**, find
**GLM 5** and **GPT OSS 120B**, review any applicable terms, and invoke each once in a playground.
This confirms availability and finishes any first-use subscription before running a full scan or
opening architecture chat.

## Runtime IAM

After model access is active, the application role/user needs only the runtime and storage access
it uses. Replace the S3 bucket name and narrow the Region wildcard if your deployment permits:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeConfiguredBedrockModels",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/zai.glm-5",
        "arn:aws:bedrock:*::foundation-model/openai.gpt-oss-120b-1:0",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
      ]
    },
    {
      "Sid": "KnowledgeObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
        "Resource": "arn:aws:s3:::beelinks-code-knowledgebase/knowledge/*"
    }
  ]
}
```

No inference-profile permission is needed for GLM-5. If a Bedrock Guardrail is configured, also
grant the guardrail permissions required by your account policy.

## Credentials

For local use, use an AWS profile, IAM Identity Center, workload credentials, or
`AWS_BEARER_TOKEN_BEDROCK`. Do not put long-lived access keys in YAML or `.env`.

For ECS, EKS, EC2, or Lambda, attach a least-privilege task, pod, instance, or execution role. No
AgentCore runtime is involved.

## S3

The configured bucket `beelinks-code-knowledgebase` was verified in `us-west-2`. Keep Block Public
Access enabled and enable versioning, default encryption, and lifecycle policies. The adapter also
requests SSE-S3 for each Markdown object. Set:

```powershell
$env:S3_BUCKET = "beelinks-code-knowledgebase"
```

Official references:

- [GLM-5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5.html)
- [GPT-OSS 120B model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html)
- [OpenAI model parameters in Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html)
- [Bedrock Runtime and Mantle endpoints](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html)
- [Bedrock structured output](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html)
- [Model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
