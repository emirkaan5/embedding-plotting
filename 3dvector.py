import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model2= SentenceTransformer('all-MiniLM-L6-v2')
openai_key = " "

from openai import OpenAI
client = OpenAI(api_key=openai_key)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)

# Example sentences and their translations
sentences_multiling = {
    "Prompt 1 en":"For two nights I had hardly had a wink of sleep, and my brain was beginning to feel that numbness which marks cerebral exhaustion. Lucy was up and in cheerful spirits. When she shook hands with me she looked sharply in my face and said:— ",
    "Prompt 1 es":"Durante dos noches apenas había podido dormir, y mi cerebro estaba comenzando a sentir ese entumecimiento que indica el agotamiento cerebral. Lucy estaba levantada y animosa. Al estrecharme la mano me miró fijamente a la cara, y dijo: ",
    "Prompt 1 tr":"İki gecedir neredeyse hiç gözümü kırpmamıştım ve beynim, yorgunluk göstergesi uyuşukluğu hissetmeye başlıyordu. Lucy uyanıktı ve neşesi yerindeydi. Benimle el sıkışırken keskin bir ifadeyle yüzüme baktı ve şöyle dedi: “Bu gece size oturmak yok. Bitkinsiniz. ",
    "Prompt 1 vi":"Sau hai đêm không ngủ, sức tôi đã giảm sút rõ rệt. Ngược lại, Lucy đã có thể đứng dậy được, tính khí của cô cũng ổn định hơn. Hai tay nắm chặt tay tôi, mắt nhìn thẳng vào mắt tôi, cô nói bằng một giọng xúc động: ",
    "random":"Outlier example",
    "random 1":"The weather is beatiful",
    "random 2 ":"Hava bugun cok iyi ya",
    "Prompt 2 en":"You want a wife to nurse and look after you a bit; that you do!” As she spoke, Lucy turned crimson, though it was only momentarily, for her poor wasted veins could not stand for long such an unwonted drain to the head. ",
    "Prompt 2 es":"Usted necesita una mujer para que le sirva de enfermera y que lo cuide un poco; ¡eso es lo que usted necesita! A medida que ella hablaba, Lucy se ruborizó, aunque sólo fue momentáneamente, pues sus pobres venas desgastadas no pudieron soportar el súbito flujo de sangre a la cabeza. ",
    "Prompt 2 tr":"Size bakacak, sizinle ilgilenecek bir eş gerek size; evet öyle!” O bunları söylerken, Lucy bir anlığına da olsa kızardı çünkü hastalıktan eriyip bitmiş damarları, başına uzun süre böylesi alışılmamış miktarda kan akıtmaya dayanamadı. ",
    "Prompt 2 vi":"Trông anh hôm nay xanh quá! Có lẽ anh cũng phải lấy vợ đi, vợ anh sẽ chú ý săn sóc anh, sẽ hết lòng vì anh, theo tôi thì đó là việc anh nên làm trong lúc này đấy! Lucy bỗng bừng đỏ mặt, các huyết mạch nghèo nàn trong người cô không còn đủ máu để cung cấp cho bộ não. ",
    "Prompt 3 en" : "Take care, for the sake of others if not for your own.” Then seeing poor Lucy scared, as she might well be, he went on more gently: “Oh, little miss, my dear, do not fear me. ",
    "Prompt 3 es" :"Cuídese, por amor a los otros si no por amor a usted misma -añadió, pero viendo que la pobre Lucy se había asustado como tenía razón de estarlo, continuó en un tono más suave-: ¡Oh, señorita, mi querida, no me tema! ",
    "Prompt 3 tr" :"Yaptığım her şeyde çetin bir amaç vardır ve sizi uyarıyorum, bana karşı çıkmayın. Kendi iyiliğiniz için olmasa bile başkalarının iyiliği için dikkat edin.” Sonra zavallı Lucy’nin, haklı olarak korktuğunu görünce, sözlerini daha nazik biçimde sürdürdü: “Ah, küçükhanım, canım, benden korkmayın. ",
    "Prompt 3 vi" :"— Em quả là một cô bé chân thành và chung thủy! Đừng khóc nữa, Lucy thân mến ạ; nếu phải lo cho tôi, thì xin em hãy đừng phiền lòng: tôi đã quen với những lúc khó khăn và lần này tôi cũng sẽ biết chịu đựng mà. ",


}

sentences_translated = {
    "Prompt 1 en": "For two nights I had barely been able to sleep, and my brain was beginning to feel that numbness that indicates brain exhaustion. Lucy was up and spirited. As he shook my hand, he looked me straight in the face and said:",
    "Prompt 1 es":"For two nights I had hardly had a wink of sleep, and my brain was beginning to feel that numbness which marks cerebral exhaustion. Lucy was up and in cheerful spirits. When she shook hands with me she looked sharply in my face and said:— ",
    "Prompt 1 tr":"I had barely slept in two nights, and my brain was starting to feel the lethargy that indicates fatigue. Lucy was awake and in good spirits. As he shook hands with me, he looked at me with a sharp expression and said: “No sitting for you tonight. You are exhausted.",
    "Prompt 1 vi":"After two nights without sleep, my strength has clearly decreased. On the contrary, Lucy was able to stand up and her temper was more stable. Holding my hand with both hands, looking straight into my eyes, she said in an emotional voice:",
    "random 0": "Outlier example",
    "random 1":"The weather is beatiful",
    "random 2":"Hava bugun cok iyi ya",
    "Prompt 2 en":"You want a wife to nurse and look after you a bit; that you do!” As she spoke, Lucy turned crimson, though it was only momentarily, for her poor wasted veins could not stand for long such an unwonted drain to the head. ",
    "Prompt 2 es":"You need a woman to serve as your nurse and take care of you a little; That's what you need! As she spoke, Lucy blushed, though it was only momentarily, for her poor worn veins could not withstand the sudden rush of blood to her head.",
    "Prompt 2 tr":"You need a spouse who will look after you and take care of you; yes like that!"" As he said this, Lucy blushed, if only for a moment, because her veins, worn out by the disease, could not bear such an unusual amount of blood flowing to her head for such a long time.",
    "Prompt 2 vi": "You look so green today! Maybe you should also get married, your wife will pay attention to take care of you, will be devoted to you, in my opinion that is what you should do at this time! Lucy suddenly blushed, the poor blood vessels in her body no longer had enough blood to supply her brain."    ,   
    "Prompt 3 vi" :"— You are such a sincere and loyal girl! Cry no more, dear Lucy; If you have to worry about me, please don't worry: I'm used to difficult times and this time I'll endure it too.",
    "Prompt 3 tr" :"There is a fierce purpose in everything I do, and I warn you, do not oppose me. Pay attention, if not for your own sake, then for the sake of others.” Then, seeing that poor Lucy was justifiably afraid, he continued more gently: “Oh, little lady, my dear, don't be afraid of me.",
    "Prompt 3 es" :"""Take care, for the love of others if not for the love of yourself,"" he added, but seeing that poor Lucy had been frightened as she had reason to be, he continued in a softer tone: ""Oh, miss, my dear, I don't care."" issue!""",
    "Prompt 3 en" : "Take care, for the sake of others if not for your own.” Then seeing poor Lucy scared, as she might well be, he went on more gently: “Oh, little miss, my dear, do not fear me. ",
   
}

# Generate embeddings
embeddings_multiling = model.encode(list(sentences_multiling.values()))
embeddings_translated = model2.encode(list(sentences_translated.values()))
embeddings_openai = [get_embedding(sentence) for sentence in sentences_multiling.values()]
dict_keys = sentences_multiling.keys()

# Reduce dimensions
pca = PCA(n_components=3)
embeddings_multiling_reduced = pca.fit_transform(embeddings_multiling)
embeddings_translated_reduced = pca.fit_transform(embeddings_translated)
embeddings_openai_reduced = pca.fit_transform(embeddings_openai)

# Plotting

fig = plt.figure(figsize=(21, 7)) 

# Plot for multilingual sentences
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(embeddings_multiling_reduced[:,0], embeddings_multiling_reduced[:,1], embeddings_multiling_reduced[:,2])
for i, key in enumerate(sentences_multiling.keys()):
    ax1.text(embeddings_multiling_reduced[i,0], embeddings_multiling_reduced[i,1], embeddings_multiling_reduced[i,2], key, color='red')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')
ax1.set_zlabel('PCA 3')
ax1.set_title('Multilingual Sentences')

# Plot for translated sentences
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(embeddings_translated_reduced[:,0], embeddings_translated_reduced[:,1], embeddings_translated_reduced[:,2])
for i, key in enumerate(sentences_translated.keys()):
    ax2.text(embeddings_translated_reduced[i,0], embeddings_translated_reduced[i,1], embeddings_translated_reduced[i,2], key, color='green')
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')
ax2.set_zlabel('PCA 3')
ax2.set_title('Translated Sentences')

# Plot for OpenAI embeddings
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(embeddings_openai_reduced[:,0], embeddings_openai_reduced[:,1], embeddings_openai_reduced[:,2])
for i, key in enumerate(sentences_multiling.keys()):  # Ensure you're using the right dataset and keys
    ax3.text(embeddings_openai_reduced[i,0], embeddings_openai_reduced[i,1], embeddings_openai_reduced[i,2], key, color='blue')
ax3.set_xlabel('PCA 1')
ax3.set_ylabel('PCA 2')
ax3.set_zlabel('PCA 3')
ax3.set_title('OpenAI Embeddings')

plt.show()
