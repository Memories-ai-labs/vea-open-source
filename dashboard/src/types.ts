export type EventType =
  | 'iteration_start'
  | 'tool_call_plan'
  | 'tool_call'
  | 'tool_result'
  | 'tool_error'
  | 'storyboard_update'
  | 'stopped_early'
  | 'paused'
  | 'resumed'
  | 'prompt_injected'
  | 'done'
  | 'error'
  | 'session_ended';

export interface PlanningEvent {
  event_type: EventType;
  data: Record<string, any>;
  timestamp: string;
}

export interface RetrievedClip {
  video_no: string;
  video_name: string;
  source_path: string;
  start_seconds: number;
  end_seconds: number;
  score: number;
  description: string;
  shot_query: string;
}

export interface Shot {
  id: string;
  purpose: string;
  search_query: string;
  retrieved_clip?: RetrievedClip;
  narration?: string;
  priority: string;
  duration_seconds: number;
}

export interface Storyboard {
  iteration: number;
  target_duration_seconds: number;
  theme: string;
  narrative_arc: string;
  shots: Shot[];
  open_questions: string[];
  notes: string;
}

export interface ToolCall {
  type: 'chat' | 'search';
  question?: string;
  query?: string;
  purpose: string;
  iteration: number;
}

export interface ProjectSummary {
  project_name: string;
  status: 'new' | 'indexed' | 'planning' | 'done' | 'error' | string;
  video_count: number;
  clip_count: number;
  iteration_count: number;
  footage_files: string[];
  indexed_files: string[];
  video_gists: Record<string, string>;
  gist: string;
  has_storyboard: boolean;
  has_fcpxml: boolean;
  has_renders: boolean;
  last_updated: string | null;
}

export interface SessionStatus {
  project_name: string;
  status: string;
  running: boolean;
  iteration: number;
  shots: number;
  clips: number;
}
