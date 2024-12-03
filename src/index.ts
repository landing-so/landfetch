import { Browser, launch, BrowserWorker, connect, sessions, ActiveSession } from '@cloudflare/puppeteer';
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

async function getRandomSession(endpoint: BrowserWorker): Promise<string | undefined> {
  const activeSessions: ActiveSession[] = await sessions(endpoint);

  const sessionIds = activeSessions.filter((session) => !session.connectionId).map((session) => session.sessionId);

  if (sessionIds.length === 0) {
    return undefined;
  }

  return sessionIds[Math.floor(Math.random() * sessionIds.length)];
}

async function getBrowser(env: Env): Promise<{ browser: Browser; launched: boolean }> {
  const sessionId = await getRandomSession(env.MYBROWSER);
  let browser: Browser | undefined;
  let launched = false;

  if (sessionId) {
    try {
      browser = await connect(env.MYBROWSER, sessionId);
      console.log(`Connected to existing session: ${sessionId}`);
    } catch (e) {
      console.log(`Failed to connect to session ${sessionId}. Error: ${e}`);
    }
  }

  if (!browser) {
    browser = await launch(env.MYBROWSER);
    launched = true;
    console.log(`Launched new session: ${browser.sessionId()}`);
  }

  return { browser, launched };
}

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
    system: `You are an expert at analyzing website structure and identifying the main website logo and favicons. 
    When looking for logos, you should ONLY identify the official logo of the website being analyzed, not any other logos that might appear in the content (like social media logos, partner logos, etc.).
    The website logo is typically found in the header area of the page and often links to the homepage.`,
    tools: {
      analyzeImages: tool({
        description: 'Analyze images and identify the main website logo and favicons',
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
    prompt: `Analyze these images from ${pageData.url} and identify the main website logo and favicons. 
    For the logo, ONLY look for the official logo of ${new URL(pageData.url).hostname} - ignore any other logos in the content.
    The website logo is typically:
    - Located in the header/top of the page
    - Links to the homepage
    - Contains the website/company name or brand
    - Has relevant alt text or class names
    
    Return two arrays:
    1. Logos array with the main website logo (if found), sorted by confidence
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

      const { browser, launched } = await getBrowser(env);
      const page = await browser.newPage();

      try {
        await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });
        const html = await page.content();
        const pageData = await extractPageData(html, targetUrl);
        const imageAnalysis = await analyzeImages(pageData, env);
        const summary = await generateSummary(pageData, env);

        const responseData = {
          meta: {
            title: pageData.title,
            description: pageData.metaDescription,
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
      } finally {
        await page.close();
        await browser.disconnect();
      }
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
