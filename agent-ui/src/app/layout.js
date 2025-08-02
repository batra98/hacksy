export const metadata = {
  title: 'Hacksy - AI Hackathon Recommender',
  description: 'AI-powered hackathon project recommendations based on your GitHub profile using Gemini AI',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0 }}>
        {children}
      </body>
    </html>
  )
}
