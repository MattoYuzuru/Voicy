from gigachat import GigaChat

from config import KEY


def create_summary(text: str) -> str:
    model = GigaChat(
        credentials=KEY,
        model="GigaChat",
        verify_ssl_certs=False
    )

    prompt = f'''
    Ты - помощник по составлению итогов текста.
    Ваша единственная задача - составить краткое изложение текста, представленного ниже. 
    Ответ должен состоять из списка ключевых фактов текста. 
    Не используй markdown, будь краток.
    Текст: {text}
    '''

    response = model.chat(prompt)

    return response.choices[0].message.content
