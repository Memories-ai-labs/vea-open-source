import os
import json
import pandas as pd
from pathlib import Path
from typing import List
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.storage_factory import get_storage_client
from src.config import BUCKET_NAME


FULL_QUIZ_QUESTIONS = [
    # Once (2007)
    {"movie": "174.曾经 Once (2007) 中英双字", "question": "What motivates the Girl to first talk to Guy, and what does she ask of him?", "answer": "The Girl is intrigued by Guy's street performance on Grafton Street and initiates a conversation by asking him to repair her broken vacuum cleaner, which sparks their creative and personal connection."},
    {"movie": "174.曾经 Once (2007) 中英双字", "question": "Through which song does Guy reveal his past heartbreak, and what does it reveal?", "answer": "Through the song 'Lies', Guy reveals that his ex-girlfriend betrayed him and left for London, highlighting his lingering emotional wounds and sense of abandonment."},
    {"movie": "174.曾经 Once (2007) 中英双字", "question": "What complicating detail about the Girl’s personal life does Guy learn?", "answer": "Guy learns that the Girl is married and has a young daughter, and although her husband is back in the Czech Republic, she still hopes to reconcile with him, complicating their budding relationship."},
    {"movie": "174.曾经 Once (2007) 中英双字", "question": "At what moment do they decide to record a demo, and why?", "answer": "After a spontaneous and emotionally charged performance of 'Falling Slowly' in a music store, they feel inspired and creatively aligned, leading them to record a demo together in a local studio."},
    {"movie": "174.曾经 Once (2007) 中英双字", "question": "Order these events correctly: (A) Girl tells Guy she's married; (B) They record a demo; (C) Guy buys her a piano.", "answer": "A → B → C."},

    # How to Train Your Dragon
    {"movie": "驯龙高手", "question": "What event causes Hiccup to meet Toothless, and what is unique about the dragon?", "answer": "Hiccup uses an experimental contraption to shoot down a mysterious Night Fury during a dragon raid. When he finds the injured dragon, he discovers that Night Furies are incredibly rare and intelligent, unlike other dragons."},
    {"movie": "驯龙高手", "question": "How does Hiccup build trust with Toothless?", "answer": "Hiccup earns Toothless’s trust through non-aggressive behavior, feeding him fish, carefully approaching him, and eventually building a prosthetic tail fin to help him fly again."},
    {"movie": "驯龙高手", "question": "What secret does Hiccup keep, and what happens when he is tested by the tribe?", "answer": "Hiccup keeps his friendship with Toothless a secret. When tested by the tribe to kill a dragon, he tries to show empathy instead of violence, but the situation escalates when Toothless appears, exposing the truth."},
    {"movie": "驯龙高手", "question": "What critical advice does Gobber give during Hiccup’s test?", "answer": "Gobber advises Hiccup to defend himself, unaware of Hiccup’s true intentions. Hiccup, however, refuses to harm the dragon, revealing his compassionate philosophy."},
    {"movie": "驯龙高手", "question": "Order these events: (A) Hiccup is asked to kill a dragon; (B) He befriends Toothless; (C) Stoick supports him in battle.", "answer": "B → A → C."},

    # V for Vendetta
    {"movie": "V字仇杀队", "question": "How does V rescue Evey and what iconic action follows this?", "answer": "V rescues Evey from an attempted assault by the secret police (Fingermen) and dramatically blows up the Old Bailey to mark the beginning of his rebellion."},
    {"movie": "V字仇杀队", "question": "Which three regime figures does V kill, and how?", "answer": "V assassinates Lewis Prothero with an overdose, poisons Bishop Lilliman with communion wine, and kills Peter Creedy in a dramatic final confrontation with knives."},
    {"movie": "V字仇杀队", "question": "How does V broadcast his message to the public?", "answer": "He takes over the British Television Network (BTN) using a disguise and pre-recorded video to deliver a revolutionary message urging citizens to rise against the government."},
    {"movie": "V字仇杀队", "question": "What does the restored underground tunnel symbolize in V’s plan?", "answer": "The tunnel beneath Parliament, filled with explosives, symbolizes the destruction of tyranny and the rebirth of a free society through revolution."},
    {"movie": "V字仇杀队", "question": "Order the following: (A) V broadcasts on BTN; (B) He destroys CCTV buildings; (C) He kills Prothero and Lilliman.", "answer": "C → A → B."},

    # Se7en
    {"movie": "七宗罪", "question": "What is John Doe’s motive behind his murders?", "answer": "John Doe commits murders based on the seven deadly sins, believing his actions are a form of divine justice and moral retribution for humanity’s corruption."},
    {"movie": "七宗罪", "question": "Which sin is represented by the first three victims, in order?", "answer": "The first three victims represent Gluttony, Greed, and Sloth, with each murder meticulously staged to highlight the sin."},
    {"movie": "七宗罪", "question": "Which detective is considering retirement, and how does that affect the investigation?", "answer": "Detective Somerset is about to retire and is cautious and methodical, while his partner Mills is impulsive, creating tension in how they handle the case."},
    {"movie": "七宗罪", "question": "What twist does the “envy” sin bring, and how does it trigger “wrath”?", "answer": "Doe confesses to being envious of Mills’s life and reveals he murdered Mills’s wife, Tracy. This provokes Mills to commit the final sin, Wrath, by killing Doe."},
    {"movie": "七宗罪", "question": "Order these events: (A) Investigation at library; (B) Discovery of church scene (greed); (C) Package arrives with head.", "answer": "A → B → C."},

    # Spirited Away
    {"movie": "千与千寻", "question": "What happens when Chihiro’s parents eat food in the tunnel town?", "answer": "They gorge on the unattended food and are transformed into pigs, trapping Chihiro in the spirit world and forcing her to find a way to save them."},
    {"movie": "千与千寻", "question": "How does Chihiro secure a job at Yubaba’s bathhouse?", "answer": "With Haku’s help, Chihiro begs Yubaba for a job and signs a contract, which steals part of her name as a condition of employment."},
    {"movie": "千与千寻", "question": "What taboo must Chihiro avoid to keep her real name?", "answer": "She must not eat the spirit world's food thoughtlessly or forget her real name, or she will be trapped in the spirit world forever."},
    {"movie": "千与千寻", "question": "How does No-Face escalate chaos, and how is Chihiro involved?", "answer": "No-Face consumes staff and offers gold to manipulate others. Chihiro confronts him, offers medicine from the river spirit, and leads him away peacefully."},
    {"movie": "千与千寻", "question": "Order these events: (A) Parents turn into pigs; (B) Chihiro meets Haku; (C) Chihiro cleans the river spirit.", "answer": "A → B → C."},

    # The Shawshank Redemption
    {"movie": "肖申克的救赎", "question": "What crime is Andy Dufresne convicted of, and what is his sentence?", "answer": "Andy is wrongly convicted of murdering his wife and her lover and is sentenced to two consecutive life terms in Shawshank Prison."},
    {"movie": "肖申克的救赎", "question": "How does Andy assist the prison administration financially?", "answer": "Andy helps the warden launder money under a fake identity while offering tax advice to guards, gaining protection and privileges."},
    {"movie": "肖申克的救赎", "question": "What role does Brooks play in illustrating institutionalization?", "answer": "Brooks is paroled after decades in prison but cannot adapt to life outside and ultimately dies by suicide, symbolizing the psychological effects of long-term incarceration."},
    {"movie": "肖申克的救赎", "question": "How does Andy escape prison, and what does he leave behind?", "answer": "Andy tunnels through the wall over 19 years, escapes through the sewage pipe, and leaves evidence of the warden’s crimes in a Bible and financial documents."},
    {"movie": "肖申克的救赎", "question": "Order the following: (A) Andy befriends Tommy; (B) Andy gets library funded; (C) Andy escapes.", "answer": "B → A → C."},

    # John Wick
    {"movie": "John Wick (2014) 1080p BluRay H264 DolbyD 5.1 + nickarad", "question": "What inciting incident pulls Wick back into action?", "answer": "John Wick is forced out of retirement after Iosef Tarasov kills his dog Daisy—gifted by his late wife—and steals his car."},
    {"movie": "John Wick (2014) 1080p BluRay H264 DolbyD 5.1 + nickarad", "question": "How does Wick learn of Iosef’s identity and father?", "answer": "The chop shop owner Aurelio identifies the car and tells Wick that Iosef is the son of Viggo Tarasov, a powerful Russian crime lord."},
    {"movie": "John Wick (2014) 1080p BluRay H264 DolbyD 5.1 + nickarad", "question": "Describe the encounter at the Red Circle nightclub.", "answer": "Wick infiltrates the Red Circle to kill Iosef. He takes down numerous guards but Iosef escapes. Wick is injured in a fight with Viggo's men."},
    {"movie": "John Wick (2014) 1080p BluRay H264 DolbyD 5.1 + nickarad", "question": "What rule does Marcus break to save Wick, and what is its consequence?", "answer": "Marcus helps Wick by shooting Viggo’s men, violating Continental neutrality. He is later tortured and executed by Viggo for his betrayal."},
    {"movie": "John Wick (2014) 1080p BluRay H264 DolbyD 5.1 + nickarad", "question": "Order these events: (A) Puppy killed; (B) Wick storms church; (C) Viggo places bounty.", "answer": "A → C → B."},

    # The Space Between Us
    {"movie": "spacebetween", "question": "What is the name of the first human born on Mars in The Space Between Us, and how was his birth kept a secret?", "answer": "Gardner Elliot is the first human born on Mars. His birth was concealed because his mother, an astronaut, died during childbirth, and the mission leaders feared public backlash and mission cancellation."},
    {"movie": "spacebetween", "question": "What medical condition prevents Gardner from living on Earth permanently?", "answer": "Gardner has an enlarged heart and a weak skeletal system due to Mars’ low gravity. Earth’s gravity puts extreme strain on his body, making long-term survival impossible."},
    {"movie": "spacebetween", "question": "What is the name of the Earth girl Gardner communicates with, and how do they first meet in person?", "answer": "Gardner communicates online with Tulsa, a girl from Colorado. After sneaking away from NASA, he meets her in person at her school."},
    {"movie": "spacebetween", "question": "What major plot twist is revealed about Gardner’s father by the end of the film?", "answer": "Gardner initially suspects a scientist is his father, but it's revealed that the mission director, Nathaniel Shepherd, is actually his biological father."}
]


class QuizingEvaluation:
    def __init__(self, benchmark_root: str = "benchmark_outputs", output_file: str = "quiz_predictions.txt"):
        self.benchmark_root = Path(benchmark_root)
        self.output_file = Path(output_file)
        self.gcs = get_storage_client()
        self.llm = GeminiGenaiManager(model="gemini-2.5-flash")
        self.questions = FULL_QUIZ_QUESTIONS
        print(f"[INIT] Loaded {len(self.questions)} hardcoded questions")
        print(f"[INIT] Benchmark root: {self.benchmark_root}")

    def _discover_models_for_movie(self, movie_name: str) -> List[str]:
        movie_dir = self.benchmark_root / movie_name
        if not movie_dir.exists():
            print(f"[WARN] Movie folder missing: {movie_dir}")
            return []
        models = [p.name for p in movie_dir.iterdir() if p.is_dir()]
        return models

    def _get_index_path(self, movie_name: str, model_name: str) -> Path:
        model_path = self.benchmark_root / movie_name / model_name
        return (
            model_path / "media_indexing.json" if model_name == "ours"
            else model_path / "naive_combined_summary.txt"
        )

    def _format_prompt(self, question: str, context: str) -> str:
        return (
            f"You are an AI tasked with answering questions about a long-form video based on provided comprehension content.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Give your best answer as if you're summarizing for a viewer. Be specific. You are not allowed to use any external information. "
            f"You should answer strictly based on the available context.\n"
        )

    def run(self):
        model_id_map = {}  # Maps model name → numeric ID
        next_model_id = 1
        output_lines = []

        for i, row in enumerate(self.questions):
            movie = row["movie"]
            question = row["question"]
            print(f"\n[QUESTION {i+1}] Movie: {movie} | Q: {question}")
            output_lines.append(f"\n=== Movie: {movie} ===\nQ{i+1}. {question}\n")

            available_models = self._discover_models_for_movie(movie)
            if not available_models:
                print(f"[SKIP] No models found for {movie}")
                continue

            for model_name in sorted(available_models):
                print(f"[MODEL] {model_name}")
                index_path = self._get_index_path(movie, model_name)
                if not index_path.exists():
                    print(f"[SKIP] Missing index for {movie} | {model_name}")
                    continue

                # Assign numeric ID
                if model_name not in model_id_map:
                    model_id_map[model_name] = str(next_model_id)
                    next_model_id += 1
                model_id = model_id_map[model_name]

                try:
                    context = (
                        index = json.load(open(index_path, "r", encoding="utf-8")) if model_name == "ours"
                        else open(index_path, "r", encoding="utf-8").read()
                    )
                    prompt = self._format_prompt(question, context)
                    predicted_answer = self.llm.LLM_request([prompt]).strip()
                    output_lines.append(f"[Model {model_id}] {predicted_answer}\n")

                except Exception as e:
                    print(f"[ERROR] Failed on {movie} | {model_name}: {e}")
                    output_lines.append(f"[Model {model_id}] ERROR: Could not generate answer.\n")

        # Write to file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\n[COMPLETE] Saved all predictions to {self.output_file}")

if __name__ == "__main__":
    evaluator = QuizingEvaluation()
    evaluator.run()
