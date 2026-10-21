/**
 * Evolving Orchestrator sub-agent tool for Pi.
 *
 * Pi ships without sub-agents. This extension adds one synchronous `agent`
 * tool: the orchestrator delegates a self-contained task, the child Pi process
 * works in the same workspace, and the orchestrator receives the final report.
 *
 * Load it for an episode:
 *
 *   pi -e crates/evolving-orchestrator/pi/subagent.ts --print --mode json ...
 *
 * Environment:
 *   EVOLVE_MODEL                 required; model id understood by Pi
 *   EVOLVE_PROVIDER              default "local"
 *   EVOLVE_LOGS                  episode logs directory; child sessions go to
 *                               <EVOLVE_LOGS>/subagents
 *   EVOLVE_PI_BIN                default "pi"
 *   EVOLVE_SUBAGENT_DEPTH        internal; starts at 0, hard limit 1
 *   EVOLVE_SUBAGENT_TIMEOUT_MS   default 900000
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MAX_DEPTH = 1;

function textFromContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";

	const parts: string[] = [];
	for (const block of content) {
		if (typeof block !== "object" || block === null) continue;
		if (!("type" in block) || !("text" in block)) continue;
		if (block.type !== "text" || typeof block.text !== "string") continue;
		parts.push(block.text);
	}
	return parts.join("");
}

async function runChild(
	args: string[],
	cwd: string,
	depth: number,
	signal: AbortSignal | undefined,
): Promise<{ stdout: string; stderr: string; code: number | null; timedOut: boolean }> {
	const timeoutMs = Number(process.env.EVOLVE_SUBAGENT_TIMEOUT_MS ?? "900000");
	const child = spawn(process.env.EVOLVE_PI_BIN ?? "pi", args, {
		cwd,
		env: { ...process.env, EVOLVE_SUBAGENT_DEPTH: String(depth) },
		stdio: ["ignore", "pipe", "pipe"],
	});

	let stdout = "";
	let stderr = "";
	let timedOut = false;

	child.stdout?.setEncoding("utf8");
	child.stderr?.setEncoding("utf8");
	child.stdout?.on("data", (chunk: string) => {
		stdout += chunk;
	});
	child.stderr?.on("data", (chunk: string) => {
		stderr += chunk;
	});

	const timer = setTimeout(() => {
		timedOut = true;
		child.kill("SIGKILL");
	}, timeoutMs);

	const abort = () => child.kill("SIGKILL");
	signal?.addEventListener("abort", abort, { once: true });

	const code = await new Promise<number | null>((resolve) => {
		child.on("close", (value) => resolve(value));
	});

	clearTimeout(timer);
	signal?.removeEventListener("abort", abort);

	return { stdout, stderr, code, timedOut };
}

function parseStream(stdout: string): {
	text: string;
	stopReason: string;
	settled: boolean;
} {
	let text = "";
	let stopReason = "";
	let settled = false;

	for (const line of stdout.split(/\r?\n/)) {
		if (!line.trim()) continue;
		let event: unknown;
		try {
			event = JSON.parse(line);
		} catch {
			continue;
		}
		if (typeof event !== "object" || event === null) continue;
		if (!("type" in event) || typeof event.type !== "string") continue;

		if (event.type === "agent_settled") {
			settled = true;
		}

		if (event.type === "message_end" && "message" in event) {
			const message = event.message;
			if (typeof message !== "object" || message === null) continue;
			if (!("role" in message) || message.role !== "assistant") continue;
			if ("content" in message) {
				const candidate = textFromContent(message.content);
				if (candidate) text = candidate;
			}
			if ("stopReason" in message && typeof message.stopReason === "string") {
				stopReason = message.stopReason;
			}
		}

		if (
			event.type === "agent_end" &&
			"messages" in event &&
			Array.isArray(event.messages)
		) {
			const messages: unknown[] = event.messages;
			for (const message of messages) {
				if (typeof message !== "object" || message === null) continue;
				if (!("role" in message) || message.role !== "assistant") continue;
				if ("content" in message) {
					const candidate = textFromContent(message.content);
					if (candidate) text = candidate;
				}
				if ("stopReason" in message && typeof message.stopReason === "string") {
					stopReason = message.stopReason;
				}
			}
		}
	}

	return { text: text.trim(), stopReason, settled };
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "agent",
			label: "Agent",
			description:
				"Delegate a self-contained task to a subagent. The child works in the same " +
				"workspace, waits for completion, and returns its final report. Use one " +
				"deliverable per call; the child cannot see this conversation.",
			parameters: Type.Object({
				role: Type.String({
					description: "The subagent's role and standing instructions.",
				}),
				task: Type.String({
					description:
						"A self-contained assignment: files to read or change, interfaces " +
						"to use, commands to run, and the report expected back.",
				}),
			}),

			async execute(_toolCallId, params, signal, _onUpdate, ctx) {
				const depth = Number(process.env.EVOLVE_SUBAGENT_DEPTH ?? "0");
				if (depth >= MAX_DEPTH) {
					return {
						content: [
							{
								type: "text",
								text: "Subagent depth limit reached. Do not delegate further; complete the task directly.",
							},
						],
						details: { error: "depth_limit", depth },
						isError: true,
					};
				}

				const model = process.env.EVOLVE_MODEL;
				if (!model) {
					throw new Error("EVOLVE_MODEL is not set; cannot start a subagent");
				}
				const provider = process.env.EVOLVE_PROVIDER ?? "local";
				const logs = process.env.EVOLVE_LOGS ?? path.join(ctx.cwd, ".evolve");
				const sessionDir = path.join(logs, "subagents");
				fs.mkdirSync(sessionDir, { recursive: true });

				const sessionId = `sub-${Date.now()}-${Math.random()
					.toString(36)
					.slice(2, 8)}`;
				const args = [
					"--print",
					"--mode",
					"json",
					"--provider",
					provider,
					"--model",
					model,
					"--system-prompt",
					params.role,
					"--approve",
					"--no-context-files",
					"--no-extensions",
					"--session-dir",
					sessionDir,
					"--session-id",
					sessionId,
					params.task,
				];

				const started = Date.now();
				const result = await runChild(args, ctx.cwd, depth + 1, signal);
				const stream = parseStream(result.stdout);
				const durationMs = Date.now() - started;
				const failed =
					result.timedOut ||
					result.code !== 0 ||
					stream.stopReason === "error" ||
					stream.stopReason === "aborted" ||
					(!stream.settled && !stream.text);

				const report =
					stream.text ||
					result.stderr.trim().slice(-4000) ||
					"(subagent produced no report)";

				return {
					content: [{ type: "text", text: report }],
					details: {
						role: params.role,
						sessionDir,
						sessionId,
						exitCode: result.code,
						durationMs,
						timedOut: result.timedOut,
						stopReason: stream.stopReason || null,
					},
					isError: failed,
				};
			},
		}),
	);
}
