import { Page, launch, BrowserWorker } from '@cloudflare/puppeteer';
import { generateText, tool } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { z } from 'zod';
import * as cheerio from 'cheerio';

export interface Env {
	MYBROWSER: BrowserWorker;
	OPENAI_API_KEY: string;
	CACHE: KVNamespace;
}

interface ImageContext {
	alt?: string;
	className?: string;
	id?: string;
	parentClasses?: string;
	nearbyText?: string;
	rel?: string;
	mimeType?: string;
}

interface Image {
	url: string;
	width?: number;
	height?: number;
	type: 'logo' | 'favicon' | 'other';
	context: ImageContext;
}

interface PageData {
	title: string;
	metaDescription: string;
	content: string;
	images: Image[];
	url: string;
}

const CACHE_TTL = 60 * 60 * 24 * 30;
const CACHE_PREFIX = 'page:';
const SESSION_TIMEOUT = 60 * 1000;
const sessions: Map<string, Page> = new Map();

async function extractPageData(html: string, baseUrl: string): Promise<PageData> {
	const $ = cheerio.load(html);
	const content = $('article, main, #content, .content, body')
		.find('p, h1, h2, h3, h4, h5, h6')
		.map((_, el) => $(el).text().trim())
		.get()
		.join('\n')
		.replace(/\s+/g, ' ')
		.trim();

	const images: Image[] = [];

	$('link[rel*="icon"]').each((_, el) => {
		const href = $(el).attr('href');
		if (href) {
			images.push({
				url: new URL(href, baseUrl).toString(),
				type: 'favicon',
				context: {
					className: $(el).attr('class'),
					id: $(el).attr('id'),
					rel: $(el).attr('rel'),
					mimeType: $(el).attr('type'),
				},
			});
		}
	});

	$('img').each((_, el) => {
		const $el = $(el);
		const src = $el.attr('src');
		if (src) {
			const $parent = $el.parent();
			images.push({
				url: new URL(src, baseUrl).toString(),
				width: parseInt($el.attr('width') || '0') || undefined,
				height: parseInt($el.attr('height') || '0') || undefined,
				type: 'other',
				context: {
					alt: $el.attr('alt'),
					className: $el.attr('class'),
					id: $el.attr('id'),
					parentClasses: $parent.attr('class'),
					nearbyText: $parent.text().trim().slice(0, 100),
				},
			});
		}
	});

	return {
		title: $('title').text().trim(),
		metaDescription: $('meta[name="description"]').attr('content') || '',
		content,
		images,
		url: baseUrl,
	};
}

async function analyzeImages(pageData: PageData, env: Env) {
	const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY });

	const imagesContext = pageData.images
		.map(
			(img, index) => `Image ${index + 1}:
		URL: ${img.url}
		Type: ${img.type}
		Size: ${img.width || 'unknown'}x${img.height || 'unknown'}
		Alt Text: ${img.context.alt || 'none'}
		Classes: ${img.context.className || 'none'}
		ID: ${img.context.id || 'none'}
		Parent Classes: ${img.context.parentClasses || 'none'}
		Nearby Text: ${img.context.nearbyText || 'none'}`
		)
		.join('\n\n');

	const { toolCalls } = await generateText({
		model: openai('gpt-4o-mini'),
		system: 'You are an expert at analyzing website structure and identifying logos and favicons.',
		tools: {
			analyzeImages: tool({
				description: 'Analyze images and identify potential logos and favicons',
				parameters: z.object({
					logos: z.array(
						z.object({
							url: z.string(),
							width: z.number().nullable(),
							height: z.number().nullable(),
							alt: z.string(),
						})
					),
					favicons: z.array(
						z.object({
							url: z.string(),
							rel: z.string(),
							type: z.string(),
						})
					),
				}),
				execute: async ({ logos, favicons }) => ({ logos, favicons }),
			}),
		},
		toolChoice: 'required',
		prompt: `Analyze these images from ${pageData.url} and identify potential logos and favicons. Return two arrays:
		1. Logos array sorted by likelihood of being a logo (most likely first)
		2. Favicons array containing favicon information

		${imagesContext}`,
		temperature: 0.1,
	});

	return toolCalls?.[0]?.args || { logos: [], favicons: [] };
}

async function generateSummary(pageData: PageData, env: Env) {
	const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY });
	const { text } = await generateText({
		model: openai('gpt-4o-mini'),
		system: 'You are a helpful assistant that creates concise summaries of web pages.',
		prompt: `Please provide a brief summary of this webpage at ${pageData.url}. Here's the content:\n\nTitle: ${pageData.title}\n\nDescription: ${pageData.metaDescription}\n\nContent: ${pageData.content}`,
		maxTokens: 200,
		temperature: 0.3,
	});
	return text;
}

async function cleanupSession(url: string) {
	const page = sessions.get(url);
	if (page) {
		try {
			await page.close();
		} catch (error) {
			console.error(`Error closing page for ${url}:`, error);
		}
		sessions.delete(url);
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);
		const targetUrl = url.searchParams.get('url');

		if (!targetUrl) {
			return new Response(JSON.stringify({ error: 'Missing url parameter' }), {
				status: 400,
				headers: { 'Content-Type': 'application/json' },
			});
		}

		try {
			const cacheKey = `${CACHE_PREFIX}${targetUrl}`;
			const cachedData = await env.CACHE.get(cacheKey);

			if (cachedData) {
				return Response.json(JSON.parse(cachedData));
			}

			const browser = await launch(env.MYBROWSER);
			let page = sessions.get(targetUrl);

			if (!page || !browser.isConnected()) {
				page = await browser.newPage();
				sessions.set(targetUrl, page);
				setTimeout(() => cleanupSession(targetUrl), SESSION_TIMEOUT);
			}

			await page.goto(targetUrl, { waitUntil: 'networkidle0' });
			const html = await page.content();
			const pageData = await extractPageData(html, targetUrl);
			const imageAnalysis = await analyzeImages(pageData, env);
			const summary = await generateSummary(pageData, env);

			const responseData = {
				meta: {
					title: pageData.title,
					description: pageData.metaDescription,
					cached: false,
					cachedAt: new Date().toISOString(),
				},
				summary,
				logos: imageAnalysis.logos,
				favicons: imageAnalysis.favicons,
			};

			await env.CACHE.put(cacheKey, JSON.stringify(responseData), {
				expirationTtl: CACHE_TTL,
			});

			return Response.json(responseData);
		} catch (error: any) {
			console.error('Error:', error);
			return new Response(
				JSON.stringify({
					error: 'Failed to fetch page data',
					details: error.message,
				}),
				{
					status: 500,
					headers: { 'Content-Type': 'application/json' },
				}
			);
		}
	},
};
