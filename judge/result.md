python -m judge.test_judge --model llama3.1:8b
8 jokes -> 28 judge calls to llama3.1:8b

Headline: Study finds cats recognize their own names but often choose to ignore them

 rank  fitted    se   joke
    1    1.37  0.99   The cats did respond instantly to one word. Researchers believe it ...
    2    1.37  0.99   In related news, my cat has read the study and declined to comment.
    3    1.37  0.99   The study took three years because the cats kept ignoring the conse...
    4    0.26  0.88   The cats reportedly responded to the findings by knocking the paper...
    5   -0.26  0.88   Finally, science confirms what my cat and my teenage son have in co...
    6   -1.37  0.99   Cats ignore their names. Cats are animals.
    7   -1.37  0.99   Cat.
    8   -1.37  0.99   Researchers also found the cats recognize the word 'study' and chos...

shown-first win rate = 12/28 = 0.43   (binomial p = 0.572; p < 0.05 means position bias)
every answer logged to llm_judgments.jsonl