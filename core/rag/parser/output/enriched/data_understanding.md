```markdown
# AeroSports Scarborough Dataset Overview

AeroSports Scarborough is a recreational facility offering various activities such as trampolines, go-karting, laser tag, and birthday party packages. This page provides an overview of the dataset comprising multiple workbooks that collectively offer comprehensive information about the business's offerings, pricing, operational details, customer service scripts, and promotional campaigns.

## General Information

**Purpose:** This workbook provides an overview of AeroSports Scarborough, including its location, contact details, operating hours, and general policies.

**Key Entities:**
- Address
- Phone number
- Operating hours
- Age restrictions
- COVID-19 safety measures
- Parking information

**Relationships:** Serves as a foundational reference for other workbooks by providing essential business details that are referenced throughout the dataset.

## Jump Prices

**Purpose:** Details the pricing structure for various jump passes, including discounts and promotional offers.

The pricing structure includes different pass types such as Premium, VIP, and Ultimate. The Premium Jump Pass lasts 90 minutes and costs $24.90. The VIP Jump Pass is available for 120 minutes at a price of $34.90. For those seeking the Ultimate experience, the Ultimate Jump Pass provides 150 minutes of fun for $44.90. Discounts are also offered, such as 25% off for groups of 15 or more. Various promotional codes can be applied to these passes.

**Relationships:** Links to the "Passes" workbook through pass descriptions and pricing details. Provides context for promotional campaigns in the "Promotions" workbook.

## Passes

**Purpose:** Describes different types of passes available, including their features and benefits.

The available pass categories are Premium, VIP, and Ultimate. Each category offers distinct features such as varying jump times and additional activities like laser tag or go-karting.

**Relationships:** Connects with "Jump Prices" for pricing details and "Promotions" for discount offers. Provides context for customer service scripts in "Voice Call Scripts."

## Promotions

**Purpose:** Outlines current promotional campaigns, including discounts and special offers.

Promotion types include birthday promotions, which offer specific discounts on party packages. The discount amounts vary depending on the promotion, with validity periods clearly defined to ensure timely use. Booking requirements must be met for these promotions to apply.

**Relationships:** References pass types from the "Passes" workbook and pricing details from "Jump Prices." Provides context for customer inquiries in "Voice Call Scripts."

## Birthday Parties

**Purpose:** Details the offerings and packages available for hosting birthday parties at AeroSports Scarborough.

Party packages are categorized as Premium, VIP, and Ultimate. Pricing per jumper varies depending on the package chosen. Inclusions typically feature pizza and drinks to enhance the party experience.

**Relationships:** Links to promotional codes in "Promotions" for discounts. Provides context for customer service interactions in "Voice Call Scripts."

## Go Karting

**Purpose:** Describes the go-karting options available, including track details and pricing.

The facility offers two types of tracks: Main and Mini. Height requirements must be met to participate in go-karting activities. Pricing is set per session, with additional costs for extended time or special events.

**Relationships:** Connects with "Passes" for add-on pricing. Provides context for customer inquiries in "Voice Call Scripts."

## Voice Call Scripts

**Purpose:** Offers structured scripts for handling customer service calls, ensuring consistent communication.

The scripts include greeting and closing phrases to maintain professionalism. Information retrieval prompts are used to gather necessary details from customers efficiently. Transfer protocols ensure smooth transitions between departments when needed.

**Relationships:** References information from all other workbooks to provide accurate responses. Ensures alignment with business policies and offerings.

## Shared Vocabulary

- **min**: "minutes" or "minimum" (context-dependent)
- **$**: "dollars"
- **AM/PM**: Time of day
- **VIP**: Very Important Pass
- **Aero Socks**: Required footwear for activities

## Cross-workbook Relationships

- **Passes and Jump Prices:** Pricing details in "Jump Prices" are linked to pass descriptions in "Passes."
- **Promotions and Birthday Parties:** Promotional codes in "Promotions" apply to packages in "Birthday Parties."
- **Go Karting and Passes:** Add-on pricing for go-karting is detailed in both "Go Karting" and "Passes."
- **Voice Call Scripts:** Integrates information from all workbooks to provide comprehensive customer service responses.

## Voicebot Considerations

- **Tables and Codes:** Simplify tables and expand cryptic codes (e.g., promo codes) for clarity.
- **Dense Markdown:** Break down dense sections into conversational language.
- **Ambiguous Pronouns:** Clarify pronouns by specifying the subject they refer to.
- **Rewriting Needs:** Heavy rewriting may be required to convert structured data into natural, engaging dialogue.
```
