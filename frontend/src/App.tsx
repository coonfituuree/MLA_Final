import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Model = 'neural_network' | 'logistic_regression' | 'word_char_logistic' | 'context_encoder'
type Vacancy = { title: string; company_profile: string; description: string; requirements: string; benefits: string }
type LearnedPrediction = { predicted_label: number; label: string; model_score: number; decision_threshold: number; model_name: Model; model_version: string; score_note: string; vocabulary_match: boolean }
type ReviewSignal = { code: string; title: string; explanation: string; method: string; evidence: { field: string; text: string; start: number; end: number }[]; source_url: string | null }
type Prediction = LearnedPrediction & { comparisons: LearnedPrediction[]; review: { version: string; method: string; signals: ReviewSignal[]; context_note?: string | null }; overall_review: { status: string; models_disagree: boolean; reasons: string[]; employer_verified: false }; input_summary: { prepared_characters: number; prepared_sha256: string; truncated: false }; bundle_version: string }
type Metrics = { precision: number; recall: number; f1: number; pr_auc_average_precision: number; test_count: number; fraudulent_count: number }
type Info = { available_models: Model[]; evaluation_scope?: string; context_encoder?: { selected: { validation: { average_precision: number } } }; review_layer: { policy: string; version: string }; model_version: string; architecture: { hidden_dim: number; input_dim: number }; scope: string; metrics: Record<string, Metrics>; training: { split: string } }
const empty: Vacancy = { title: '', company_profile: '', description: '', requirements: '', benefits: '' }
const names: Record<string, string> = { neural_network: 'Neural network', logistic_regression: 'Logistic Regression', majority_baseline: 'Majority baseline', context_encoder: 'Compact text encoder', word_char_logistic: 'Word + character Logistic Regression' }
const percent = (n: number) => `${(n * 100).toFixed(1)}%`
async function readResponse(response: Response) {
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((e: { loc: string[]; msg: string }) => `${e.loc.slice(1).join(' ')}: ${e.msg}`).join(' · ') : `Request failed (${response.status}). Check that the backend is running.`)
  }
  if (body && 'model_score' in body && !body.overall_review) throw new Error('The backend is running an earlier application version. Stop it and restart ./scripts/start.sh to load the updated review API.')
  return body
}
function Shield({ small = false }: { small?: boolean }) {
  return <svg aria-hidden="true" width={small ? '20' : '30'} height={small ? '20' : '30'} viewBox="0 0 24 24" fill="none"><path d="M12 3 4 6v6c0 4 4 7 8 9 4-2 8-5 8-9V6l-8-3Z" stroke="currentColor" strokeWidth="1.5"/><path d="m8.5 12 2.5 2.5 4.5-5" stroke="currentColor" strokeWidth="1.5"/></svg>
}
function App() {
  const [vacancy, setVacancy] = useState<Vacancy>(empty)
  const [mode, setMode] = useState<'text' | 'file'>('text')
  const [model, setModel] = useState<Model>('neural_network')
  const [laterInstructions, setLaterInstructions] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState<Info | null>(null)
  const [serviceError, setServiceError] = useState('')
  const [result, setResult] = useState<Prediction | null>(null)
  const [stale, setStale] = useState(false)
  const resultRef = useRef<HTMLDivElement>(null)
  const inFlight = useRef(false)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/v1/model-info', { signal: controller.signal }).then(readResponse).then(setInfo).catch(e => {
      if (e.name !== 'AbortError') setServiceError(e.message)
    })
    return () => controller.abort()
  }, [])
  const update = (field: keyof Vacancy, value: string) => { setVacancy(prev => ({ ...prev, [field]: value })); setStale(true); setError('') }
  async function analyze(event: FormEvent) {
    event.preventDefault()
    if (inFlight.current) return
    setError('')
    if (mode === 'text' && (!vacancy.description.trim() || Object.values(vacancy).join('').length > 60000)) { setError('Enter a description and keep the total input under 60,000 characters.'); return }
    if (mode === 'file' && (!file || !file.name.toLowerCase().endsWith('.txt') || file.size > 120000)) { setError('Choose a UTF-8 .txt file up to 120,000 bytes.'); return }
    setBusy(true); inFlight.current = true
    try {
      let response: Response
      const signal = AbortSignal.timeout(30000)
      if (mode === 'file' && file) {
        const data = new FormData(); data.append('file', file); data.append('model', model); data.append('later_instructions', laterInstructions)
        response = await fetch('/api/v1/predictions/file', { method: 'POST', body: data, signal })
      } else {
        response = await fetch('/api/v1/predictions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...vacancy, later_instructions: laterInstructions, model }), signal })
      }
      setResult(await readResponse(response)); setStale(false)
      setTimeout(() => resultRef.current?.focus(), 0)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not analyze this advertisement.') }
    finally { setBusy(false); inFlight.current = false }
  }
  return <>
    <header className="topbar"><a className="brand" href="#"><span className="brand-icon"><Shield small /></span>VacancyLens<span className="brand-tag">RESEARCH LAB</span></a><span className={`service ${info ? 'online' : ''}`}><i />{info ? 'Local models ready' : serviceError ? 'Models unavailable' : 'Connecting to models'}</span></header>
    <main>
      <section className="intro"><div><div className="eyebrow">JOB ADVERTISEMENT SCREENING</div><h1>A closer look.<br /><span>A more informed next step.</span></h1><p>Look for patterns associated with recruitment fraud. Paste an English job advertisement and let a locally trained model help you decide what deserves a second look.</p></div><div className="intro-note"><Shield /><div><strong>Your text stays local.</strong><p>No external classifier. No saved uploads.<br />Analysis runs on your machine.</p></div></div></section>
      {serviceError && <div className="error" role="alert">{serviceError}</div>}
      <div className="workspace">
        <section className="card input-card" aria-labelledby="input-title"><div className="card-heading"><div><div className="eyebrow">01 / THE ADVERTISEMENT</div><h2 id="input-title">What opportunity are you reviewing?</h2></div></div>
          <form onSubmit={analyze}>
            <fieldset disabled={busy} className="input-fields">
              <div className="tabs" role="group" aria-label="Input method"><button type="button" aria-pressed={mode === 'text'} className={mode === 'text' ? 'active' : ''} onClick={() => { setMode('text'); setStale(true); setError('') }}>Paste advertisement</button><button type="button" aria-pressed={mode === 'file'} className={mode === 'file' ? 'active' : ''} onClick={() => { setMode('file'); setStale(true); setError('') }}>Upload .txt file</button></div>
              {mode === 'text' ? <div className="fields"><label htmlFor="title">Job title <span>Optional</span></label><input id="title" placeholder="e.g. Marketing Coordinator" value={vacancy.title} maxLength={500} onChange={e => update('title', e.target.value)} />
                <label htmlFor="description">Job description <span>Required</span></label><textarea id="description" required rows={9} maxLength={60000} placeholder="Paste the original advertisement here. Include the responsibilities, hiring process, and any details about the opportunity…" value={vacancy.description} onChange={e => update('description', e.target.value)} /><div className="field-hint">Use the original English text. More context helps. <span>{vacancy.description.length.toLocaleString()} characters</span></div>
                <details><summary>Add more context <span>Company, requirements & benefits</span></summary><div className="optional-fields">{(['company_profile', 'requirements', 'benefits'] as const).map(field => <div key={field}><label htmlFor={field}>{field === 'company_profile' ? 'Company profile' : field === 'requirements' ? 'Requirements' : 'Benefits'} <span>Optional</span></label><textarea id={field} rows={3} maxLength={20000} value={vacancy[field]} onChange={e => update(field, e.target.value)} /></div>)}</div></details>
              </div> : <div className="upload"><span aria-hidden="true" className="upload-symbol">↥</span><h3>Bring the full advertisement</h3><p>One UTF-8 .txt file, up to 120 KB.<br />Its contents are used as the description.</p><label className="file-label" htmlFor="file">Choose text file</label><input id="file" type="file" accept=".txt,text/plain" onChange={e => { setFile(e.target.files?.[0] ?? null); setStale(true); setError('') }} /><div className="filename">{file ? file.name : 'No file selected'}</div></div>}
              <details className="later-context"><summary>Did the recruiter send different instructions?</summary><p className="context-help">Optional. Paste later instructions to review alongside the advertisement. This text is checked only by the separate heuristic layer; it does not change the learned fraud scores.</p><label htmlFor="later-instructions">Later recruiter instructions</label><textarea id="later-instructions" maxLength={20000} rows={4} value={laterInstructions} onChange={e => { setLaterInstructions(e.target.value); setStale(true); setError('') }} /></details>
              <div className="model-row"><div><label htmlFor="model">Analysis model</label><p>Compare trained models; scores have different scales.</p></div><select id="model" value={model} onChange={e => { setModel(e.target.value as Model); setStale(true) }}><option value="neural_network">Neural network · default</option><option value="logistic_regression">Logistic Regression · original</option>{info?.available_models?.includes('word_char_logistic') && <option value="word_char_logistic">Word + character LR · experimental</option>}{info?.available_models?.includes('context_encoder') && <option value="context_encoder">Compact text encoder · experimental</option>}</select></div>
              {error && <div className="error" role="alert">{error}</div>}
              <button className="analyze" type="submit" disabled={busy || !info}>{busy ? <><span className="spinner" />Analyzing advertisement…</> : <>Analyze advertisement <span aria-hidden="true">↗</span></>}</button>
              <p className="privacy">Processed locally. Uploaded text is discarded after analysis.</p>
            </fieldset>
          </form>
        </section>
        <aside>
          <section className={`card result-card ${result?.overall_review.status === 'additional_review_recommended' ? 'flagged' : ''}`} ref={resultRef} tabIndex={-1} aria-labelledby="result-title" aria-live="polite" aria-busy={busy}><div className="eyebrow">02 / THE REVIEW</div>
            {!result ? <><div className="empty-icon"><Shield /></div><h2 id="result-title">A little context.<br />A clearer perspective.</h2><p>Your result will appear here once you analyze an advertisement.</p><div className="empty-steps"><div><span>1</span>Add the original vacancy text</div><div><span>2</span>Choose a trained model</div><div><span>3</span>Review the separate signals</div></div></> : <>
              <div className="result-tag">{result.overall_review.status === 'additional_review_recommended' ? 'REVIEW RECOMMENDED' : 'NO ADDITIONAL SIGNAL FOUND'}</div>
              <h2 id="result-title">{result.overall_review.status === 'additional_review_recommended' ? 'Additional review recommended' : 'No additional signal found'}</h2>
              <p>{result.overall_review.status === 'additional_review_recommended' ? 'Review the learned prediction and any separate text warnings below. These signals do not establish fraud or illegal activity.' : 'These checks found no additional warning. The employer and the opportunity have not been verified.'}</p>
              <div className="learned-result"><h3>Learned fraud classification</h3><p data-testid="learned-classification">{names[result.model_name]}: <strong>{result.predicted_label === 1 ? 'Potentially fraudulent' : 'Predicted legitimate'}</strong></p><div className="score-label"><span>Learned score</span><strong>{result.model_score.toFixed(3)}</strong></div><div className="score-track" aria-hidden="true"><div style={{ width: percent(result.model_score) }} /><i style={{ left: percent(result.decision_threshold) }} /></div><div className="scale"><span>0.000</span><span>Threshold {result.decision_threshold.toFixed(3)}</span><span>1.000</span></div><p className="score-note">Uncalibrated score, not a verified probability.</p></div>
              {result.overall_review.models_disagree && <div className="disagreement" role="note"><strong>The primary models disagree.</strong><p>One crosses its own review threshold and the other does not. Their scores are not comparable probabilities, and neither result establishes which model is correct.</p></div>}
              <div className="review-signals"><h3>Additional text review <span>HEURISTIC</span></h3><p className="review-caption">Separate pattern checks, not neural-network explanations.</p>{result.review.context_note && <p className="context-help">{result.review.context_note}</p>}{result.review.signals.length === 0 ? <p>No contextual heuristic warning found. This does not establish safety.</p> : result.review.signals.map(signal => <article className="review-signal" key={signal.code + signal.evidence[0]?.field}><h4>{signal.title}</h4>{signal.evidence.map((e, i) => <blockquote key={i}><span>{e.field.replaceAll('_',' ')}</span>{e.text}</blockquote>)}<p>{signal.explanation}</p>{signal.source_url && <a href={signal.source_url} target="_blank" rel="noreferrer">Background guidance ↗</a>}</article>)}</div>
              <details className="technical"><summary>Model comparison & technical details</summary><p>Each model uses the same complete advertisement. Later instructions are reviewed separately.</p><div className="comparison-list">{result.comparisons.map(item => <div key={item.model_name}><strong>{names[item.model_name]}</strong><span>{item.predicted_label ? 'Potentially fraudulent' : 'Predicted legitimate'}</span><span>Score {item.model_score.toFixed(4)} / threshold {item.decision_threshold.toFixed(4)}</span><small>{item.model_version}</small></div>)}</div><p>No scores are averaged. A review is recommended when a primary model or the selected model crosses its own threshold, or when a contextual heuristic fires.</p><dl><div><dt>Bundle</dt><dd>{result.bundle_version}</dd></div><div><dt>Heuristic version</dt><dd>{result.review.version}</dd></div><div><dt>Processed advertisement</dt><dd>{result.input_summary.prepared_characters.toLocaleString()} characters · no truncation</dd></div></dl><p className="input-hash">Input digest: {result.input_summary.prepared_sha256}</p></details>
              {!result.vocabulary_match && <p className="error">No known vocabulary was found. This result has very limited value.</p>}{stale && <p className="stale">Input or model changed. Analyze again to update this result.</p>}
            </>}
          </section>
          <section className="caution"><span aria-hidden="true">i</span><div><strong>A signal, not a verdict.</strong><p>This tool does not verify an employer. Independently check the company and hiring process before sharing personal information or making payments.</p></div></section>
        </aside>
      </div>
      <section className="card evidence" aria-labelledby="evidence-title"><div className="evidence-header"><div><div className="eyebrow">BEHIND THE MODEL</div><h2 id="evidence-title">Measured performance. Defined limits.</h2><p>Fixed historical benchmark, with validation-selected thresholds. This benchmark was previously inspected; it is not a new independent evaluation.</p></div><span className="outline-tag">EMSCAD / ENGLISH</span></div>
        {info ? <><div className="table-scroll"><table><caption className="sr-only">Fraudulent-class metrics on the held-out test partition</caption><thead><tr><th scope="col">Model</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">PR-AUC (AP)</th></tr></thead><tbody>{['neural_network','logistic_regression', ...(info.metrics.word_char_logistic ? ['word_char_logistic'] : []), ...(info.metrics.context_encoder ? ['context_encoder'] : []), 'majority_baseline'].map(key => { const m = info.metrics[key]; return <tr key={key}><th scope="row">{names[key]} {key === 'neural_network' && <small>DEFAULT</small>}</th><td>{percent(m.precision)}</td><td>{percent(m.recall)}</td><td>{m.f1.toFixed(3)}</td><td>{m.pr_auc_average_precision.toFixed(3)}</td></tr> })}</tbody></table></div><div className="evidence-notes"><p><strong>The evaluation</strong>{info.metrics.neural_network.test_count.toLocaleString()} test advertisements, including {info.metrics.neural_network.fraudulent_count} fraudulent examples. Duplicate and shared employer groups stay within one partition. PR-AUC is scikit-learn average precision.</p><p><strong>The boundaries</strong>No illegal-recruitment labels were used. Additional text review is heuristic. 17,880 original advertisements from 2012–2014. Historical language, platform, and geography biases limit use on current vacancies. Class-weighted scores have not been calibrated.</p><p><strong>The original neural network</strong>{info.architecture.input_dim.toLocaleString()} TF-IDF features → {info.architecture.hidden_dim} ReLU units → dropout → one logit. Trained locally with PyTorch; no external LLM classifier.</p></div></> : <p className="info-placeholder">{serviceError ? 'Evaluation results are unavailable until the trained model bundle loads.' : 'Loading actual model evaluation…'}</p>}
      </section>
      <footer><span className="footer-brand">VacancyLens</span><span>University ML final project · Local research application</span><a href="https://doi.org/10.3390/fi9010006" target="_blank" rel="noreferrer">Dataset research ↗</a></footer>
    </main>
  </>
}
export default App
